# =============================
# Librerías
# =============================

#Manejo de datos y gráficos
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")  # backend interactivo para mostrar ventanas de gráficos
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Validación/limpiar, parsear JSON y salida
import ast
import io
import contextlib
import re
import json

# Predicciones de series de tiempo
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# Conexión con Ollama y el modeolo

import ollama
import subprocess

llm = "qwen2.5:7b"  # Modelo (se puede cambiar por llama, mistral, qwen, etc.) mistral:7b llama3.2:3b  qwen2.5:14b qwen3:8b
subprocess.run(["ollama", "pull", llm], check=False) #se hace el pull del modelo si no está descargado


# =============================
# Herramienta para leer CSV y guardar en dtf
# =============================
def load_csv(path: str) -> str:
    global dtf # Variable global para almacenar el DataFrame
    try:
        dtf = pd.read_csv(path) 

        # Detectar automáticamente la columna de fechas
        datetime_col = None
        for col in dtf.columns:
            if "fecha" in col.lower() or "date" in col.lower() or "time" in col.lower():
                datetime_col = col
                break

        if datetime_col is None:
            return f"Error: no se encontró ninguna columna de fechas en {path}. Columnas: {list(dtf.columns)}"

        # Convertir a datetime y usar como índice
        dtf[datetime_col] = pd.to_datetime(dtf[datetime_col])
        dtf.set_index(datetime_col, inplace=True)

        print(dtf.head()) # Mostrar las primeras filas del DataFrame cargado
        return f"Archivo {path} cargado correctamente con {dtf.shape[0]} filas. Índice temporal: {datetime_col}. Columnas: {list(dtf.columns)}"
    except Exception as e:
        return f"Error al cargar el archivo: {e}"

# Definir la herramienta load_csv para que Ollama pueda usarla y que parámetros acepta
tool_load_csv = {
  'type': 'function',
  'function': {
    'name': 'load_csv',
    'description': 'Carga un archivo CSV con series de tiempo de energía. El DataFrame se llama dtf.',
    'parameters': {
      'type': 'object',
      'required': ['path'],
      'properties': {
        'path': {'type':'string', 'description':'ruta del archivo CSV a cargar'}
      }
    }
  }
}

# =============================
# Herramienta para respuesta final
# Sirve para devolver una respuesta en lenguaje natural al usuario
# =============================
def final_answer(text:str) -> str:
    return text

tool_final_answer = {
  'type': 'function',
  'function': {
    'name': 'final_answer',
    'description': 'Devuelve una respuesta en lenguaje natural al usuario',
    'parameters': {
      'type': 'object',
      'required': ['text'],
      'properties': {
        'text': {'type':'string', 'description':'respuesta en lenguaje natural'}
      }
    }
  }
}

# =============================
# Herramienta para ejecutar código
# Esta tool esta para arreglar errores de sintaxis comunes del modelo
# =============================


def is_valid_python(code: str) -> bool:
    """Verifica si el código es sintácticamente válido en Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False





def sanitize_code(code: str) -> str:
    code = code.replace("df[", "dtf[")

    # Si detecta un acceso incompleto 
    if re.match(r'^\s*print\s*\(\s*dtf\[\s*$', code):
     return "Código incompleto, debes especificar la columna y la operación"


    # autocierre de paréntesis, corchetes y comillas
    if code.count("(") > code.count(")"):
        code += ")" * (code.count("(") - code.count(")"))
    if code.count("[") > code.count("]"):
        code += "]" * (code.count("[") - code.count("]"))
    if code.count('"') % 2 != 0:
        code += '"'
    if code.count("'") % 2 != 0:
        code += "'"

    return code



def code_exec(code: str) -> str:
    
    output = io.StringIO()
    retries = 0
    code = code.strip()

    while retries < 10:
        code = sanitize_code(code)

        # Forzar que sea un print(...)
        if not code.startswith("print(") or not code.endswith(")"):
            return "Error: el código debe ser una sola instrucción print() cerrada."

        if is_valid_python(code):
            break
        retries += 1

    if not is_valid_python(code):
        return "Error: el código está incompleto. Debes usar algo como print(dtf[].max())."


    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            print(f"Error: {e}")

    return output.getvalue()


tool_code_exec = {
  'type':'function',
  'function':{
    'name': 'code_exec',
    'description': 'Ejecuta código Python. Siempre usar print() para mostrar la salida.',
    'parameters': {
      'type': 'object', 
      'required': ['code'],
      'properties': {
        'code': {'type':'str', 'description':'código Python a ejecutar'},
      }
    }
  }
}


# =============================
# Herramienta para graficar datos
# =============================
def normalize_plot_args(t_inputs):
    """
    Normaliza los argumentos para plot_data y corrige errores comunes del modelo.
    """
    if not isinstance(t_inputs, dict):
        return t_inputs
    
    # Asegurar que "columns" sea lista
    if "columns" in t_inputs:
        if isinstance(t_inputs["columns"], str):
            try:
                # Convierte "['MW']" → ["MW"]
                t_inputs["columns"] = json.loads(t_inputs["columns"].replace("'", '"'))
            except Exception:
                # Si falla, convierte a lista simple
                t_inputs["columns"] = [t_inputs["columns"]]
        elif t_inputs["columns"] is None:
            t_inputs["columns"] = []
    
    return t_inputs


def plot_data(columns=None, start_date=None, end_date=None, title="Gráfico de datos"):
    """
    Genera un gráfico con matplotlib a partir del DataFrame dtf.
    columns: lista de columnas a graficar. Si es None, grafica todas.
    start_date, end_date: rango de fechas opcional.
    title: título del gráfico que se puede ajustar según el contexto.
    """
    try:
        df = dtf.copy()

        # Validar columnas
        if columns:
            columnas_validas = [c for c in columns if c in df.columns]
            if not columnas_validas:
                return f"Error: ninguna de las columnas {columns} existe en dtf. Columnas disponibles: {list(df.columns)}"
            df = df[columnas_validas]

        # Filtrar por fechas
        if start_date and end_date:
            if start_date == end_date:
                # Filtrar solo el día exacto
                df = df.loc[start_date]
            else:
                # Filtrar rango de fechas
                df = df.loc[start_date:end_date]
        elif start_date:
            # Filtrar un único día
            df = df.loc[start_date]

        # Graficar
        df.plot(figsize=(12,5), linestyle="--")
        plt.title(title)
        plt.xlabel("Índice (ej: tiempo o filas)")
        plt.ylabel("Valores")
        plt.grid(True)
        plt.show()

        return f"Gráfico generado con columnas {columns or list(dtf.columns)}."
    except Exception as e:
        return f"Error al graficar: {e}"


tool_plot_data = {
  'type': 'function',
  'function': {
    'name': 'plot_data',
    'description': 'Genera un gráfico del dataset dtf usando matplotlib.',
    'parameters': {
      'type': 'object',
      'properties': {
        'columns': {
          'type': 'array',
          'items': {'type': 'string'},
          'description': 'Lista de columnas a graficar. Si no se da, se grafican todas.'
        },
        'start_date': {
          'type':'string',
          'description':'Fecha de inicio en formato YYYY-MM-DD (opcional)'
        },
        'end_date': {
          'type':'string',
          'description':'Fecha de fin en formato YYYY-MM-DD (opcional)'
        },
        'title': {
          'type':'string',
          'description':'Título del gráfico'
        }
      }
    }
  }
}


def normalize_csv_args(t_inputs):
    """
    Normaliza los argumentos que llegan a load_csv para que siempre terminen
    como {"path": "archivo.csv"} válido.
    Si no se puede normalizar, devuelve None.
    """
    # Caso directo: {"path": "archivo.csv"}
    if isinstance(t_inputs, dict) and "path" in t_inputs and isinstance(t_inputs["path"], str):
        candidate = t_inputs["path"].strip()
        # Caso especial: viene como "{'path': 'datos_limpios.csv'}"
        if candidate.startswith("{") and "path" in candidate:
            try:
                inner = json.loads(candidate.replace("'", '"'))
                return {"path": inner["path"]}
            except Exception:
                return None
        # Caso válido normal
        if candidate not in ["{", "}", ")", ""]:
            return {"path": candidate}
        return None

    # Caso {"path": {"value": "archivo.csv"}} o similar
    if isinstance(t_inputs, dict) and "path" in t_inputs and isinstance(t_inputs["path"], dict):
        inner = t_inputs["path"]
        if "value" in inner:
            return {"path": inner["value"]}
        if "file_path" in inner:
            return {"path": inner["file_path"]}
        if "path" in inner:
            return {"path": inner["path"]}

    # Caso string : "archivo.csv"
    if isinstance(t_inputs, str):
        candidate = t_inputs.strip().strip("{}()")
        if candidate != "":
            return {"path": candidate}
        return None

    # Si no se reconoce → error
    return None



# =============================
# Herramienta para predicciones de series de tiempo
# =============================


def horizon_to_periods(index, horizon_days):
    """
    Convierte un horizonte en días al número de pasos (filas) y a la frecuencia
    real de los datos, para que "horizon" siempre signifique días de calendario
    sin importar la granularidad del dataset cargado (15min, horaria, diaria, etc.).
    Devuelve (periods, freq).
    """
    freq = pd.infer_freq(index) or "D"
    try:
        delta = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
        if delta <= pd.Timedelta(0):
            return horizon_days, freq
        periods = max(int(round(pd.Timedelta(days=horizon_days) / delta)), 1)
        return periods, freq
    except Exception:
        # Frecuencias no fijas (ej. "M", "MS") no se pueden convertir a Timedelta
        return horizon_days, freq


def seasonal_period_for_freq(freq):
    """
    Estima un periodo estacional razonable ("m") para SARIMA según la frecuencia
    de los datos: ciclo diario para datos intradía (ej. 96 pasos para 15min),
    ciclo semanal para datos diarios. Devuelve 0 si no aplica estacionalidad simple.
    """
    try:
        delta = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except Exception:
        return 0
    if delta <= pd.Timedelta(0):
        return 0
    day = pd.Timedelta(days=1)
    if delta < day:
        m = int(round(day / delta))
        return m if m > 1 else 0
    if delta == day:
        return 7  # ciclo semanal para datos diarios
    return 0  # frecuencias mayores (semanal, mensual, ...): sin estacionalidad simple


def fit_sarima_forecast(serie, freq, periods):
    """
    Ajusta un SARIMAX con estacionalidad inferida de la frecuencia de los datos
    y devuelve el forecast para "periods" pasos hacia adelante.
    Trunca el set de entrenamiento cuando el periodo estacional es grande
    (ej. 96 para datos de 15min) para mantener el ajuste rápido.
    """
    m = seasonal_period_for_freq(freq)
    max_train = m * 30 if m > 1 else 3000
    train_s = serie.tail(max_train) if len(serie) > max_train else serie
    seasonal_order = (1, 0, 1, m) if m > 1 else (0, 0, 0, 0)
    fit = SARIMAX(train_s, order=(2, 1, 2), seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    forecast = fit.forecast(steps=periods)
    return forecast


def predict_data(model="prophet", column=None, horizon=7):
    """
    Genera predicciones de series de tiempo usando Prophet o ARIMA y grafica los valores futuros.
    - model: "prophet" o "arima"
    - column: nombre de la columna a predecir 
    - horizon: horizonte de predicción en días
    """
    try:
        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: debes especificar una columna válida. Columnas disponibles: {list(df.columns)}"

        df = df[[column]].dropna().reset_index()
        df.columns = ["ds", "y"]  # Prophet requiere estos nombres

        periods, freq = horizon_to_periods(pd.DatetimeIndex(df["ds"]), horizon)

        if model.lower() == "prophet":
            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            forecast = m.predict(future)

            # Graficar
            fig, ax = plt.subplots(figsize=(10, 5))
            m.plot(forecast, ax=ax)
            plt.title(f"Predicción con Prophet para {column} ({horizon} días)")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.grid(True)
            plt.show()

            # Devolver últimos valores predichos
            tail = forecast[["ds", "yhat"]].tail(periods)
            return f"Predicción con Prophet completada. Últimos valores:\n{tail.to_string(index=False)}"

        elif model.lower() == "arima":
            df.set_index("ds", inplace=True)
            try:
                df.index.freq = pd.tseries.frequencies.to_offset(freq)  # evita warning de frecuencia no informada
            except ValueError:
                pass  # índice no perfectamente regular; se deja que ARIMA infiera igual
            model_fit = ARIMA(df["y"], order=(2, 1, 2)).fit()
            forecast = model_fit.forecast(steps=periods)
            future_dates = pd.date_range(df.index[-1], periods=periods + 1, freq=freq)[1:]

            # Graficar
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["y"], label="Datos reales")
            plt.plot(future_dates, forecast, label="Predicción", linestyle="--")
            plt.title(f"Predicción con ARIMA para {column} ({horizon} días)")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            plt.show(block=True)
            plt.show()

            # Devolver últimos valores predichos
            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": forecast.values
            })
            return f"Predicción con ARIMA completada. Últimos valores:\n{forecast_df.to_string(index=False)}"

        elif model.lower() == "sarima":
            df.set_index("ds", inplace=True)
            forecast = fit_sarima_forecast(df["y"], freq, periods)
            future_dates = pd.date_range(df.index[-1], periods=periods + 1, freq=freq)[1:]

            # Graficar
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df["y"], label="Datos reales")
            plt.plot(future_dates, forecast, label="Predicción", linestyle="--")
            plt.title(f"Predicción con SARIMA para {column} ({horizon} días)")
            plt.xlabel("Fecha")
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            plt.show()

            # Devolver últimos valores predichos
            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": forecast.values
            })
            return f"Predicción con SARIMA completada. Últimos valores:\n{forecast_df.to_string(index=False)}"

        else:
            return "Error: modelo no reconocido. Usa 'prophet', 'arima' o 'sarima'."

    except Exception as e:
        return f"Error durante la predicción: {e}"


tool_predict_data = {
  'type': 'function',
  'function': {
    'name': 'predict_data',
    'description': 'Genera predicciones de series de tiempo con Prophet, ARIMA o SARIMA. Siempre grafica los valores futuros junto con los datos históricos y devuelve un resumen de los últimos valores predichos.',
    'parameters': {
      'type': 'object',
      'properties': {
        'model': {
          'type': 'string',
          'description': 'Modelo de predicción a usar: "prophet", "arima" o "sarima". Usa "sarima" cuando el usuario pida explícitamente SARIMA o capturar estacionalidad (ciclo diario/semanal).'
        },
        'column': {
          'type': 'string',
          'description': 'Nombre de la columna a predecir'
        },
        'horizon': {
          'type': 'integer',
          'description': 'Horizonte de predicción en días.'
        }
      },
      'required': ['model', 'column']
    }
  }
}

# =============================
# Herramienta para predicción con KNN
# =============================


def predict_knn(column=None, horizon=7, n_neighbors=5, n_lags=7):
    """
    Genera predicciones de series de tiempo usando KNeighborsRegressor.
    Construye features de rezago (lags) y predice paso a paso (recursivo).
    - column: nombre de la columna a predecir
    - horizon: horizonte de predicción en días
    - n_neighbors: número de vecinos a usar en KNN
    - n_lags: número de rezagos usados como features
    """
    try:
        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: debes especificar una columna válida. Columnas disponibles: {list(df.columns)}"

        serie = df[column].dropna()
        if len(serie) <= n_lags:
            return f"Error: no hay suficientes datos ({len(serie)}) para usar {n_lags} rezagos."

        periods, freq = horizon_to_periods(serie.index, horizon)

        values = serie.values

        # Construir matriz de features (lags) y target
        X, y = [], []
        for i in range(n_lags, len(values)):
            X.append(values[i - n_lags:i])
            y.append(values[i])
        X, y = np.array(X), np.array(y)

        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X, y)

        # Predicción recursiva: usa las predicciones previas para predecir el siguiente paso
        last_window = list(values[-n_lags:])
        preds = []
        for _ in range(periods):
            x_input = np.array(last_window[-n_lags:]).reshape(1, -1)
            next_val = model.predict(x_input)[0]
            preds.append(next_val)
            last_window.append(next_val)

        future_dates = pd.date_range(serie.index[-1], periods=periods + 1, freq=freq)[1:]
        forecast_df = pd.DataFrame({"ds": future_dates, "yhat": preds})

        # Graficar
        plt.figure(figsize=(10, 5))
        plt.plot(serie.index, serie.values, label="Datos reales")
        plt.plot(future_dates, preds, label="Predicción KNN", linestyle="--")
        plt.title(f"Predicción con KNN para {column} ({horizon} días)")
        plt.xlabel("Fecha")
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.show()

        return f"Predicción con KNN completada. Últimos valores:\n{forecast_df.to_string(index=False)}"

    except Exception as e:
        return f"Error durante la predicción con KNN: {e}"


tool_predict_knn = {
  'type': 'function',
  'function': {
    'name': 'predict_knn',
    'description': 'Genera predicciones de series de tiempo usando KNeighborsRegressor (KNN) con features de rezago. Siempre grafica los valores futuros junto con los datos históricos y devuelve un resumen de los últimos valores predichos.',
    'parameters': {
      'type': 'object',
      'properties': {
        'column': {
          'type': 'string',
          'description': 'Nombre de la columna a predecir'
        },
        'horizon': {
          'type': 'integer',
          'description': 'Horizonte de predicción en días.'
        },
        'n_neighbors': {
          'type': 'integer',
          'description': 'Número de vecinos a usar en KNN (por defecto 5).'
        },
        'n_lags': {
          'type': 'integer',
          'description': 'Número de rezagos (valores pasados) usados como features (por defecto 7).'
        }
      },
      'required': ['column']
    }
  }
}


# =============================
# Herramienta para comparar modelos (dashboard de predicciones)
# =============================


def compare_models(column=None, horizon=7, models=None, n_neighbors=5, n_lags=7):
    """
    Compara predicciones de prophet, arima, sarima y/o knn contra datos reales retenidos
    (holdout) y muestra un gráfico interactivo de Plotly con las métricas de error (MAE, RMSE, MAPE).
    - column: nombre de la columna a predecir y comparar
    - horizon: días retenidos como datos reales de prueba (holdout) y horizonte de predicción
    - models: subconjunto de ["prophet","arima","sarima","knn"]. Si es None, se comparan
      prophet, arima y knn (sarima es más lento de ajustar, por lo que solo se incluye si
      se solicita explícitamente).
    - n_neighbors, n_lags: hiperparámetros de KNN (mismos defaults que predict_knn)
    """
    try:
        if not models:
            models = ["prophet", "arima", "knn"]

        df = dtf.copy()
        if column is None or column not in df.columns:
            return f"Error: debes especificar una columna válida. Columnas disponibles: {list(df.columns)}"

        serie = df[column].dropna()
        periods, freq = horizon_to_periods(serie.index, horizon)
        if len(serie) <= periods:
            return f"Error: la serie tiene {len(serie)} filas, insuficientes para un holdout de {horizon} días ({periods} pasos)."

        train = serie.iloc[:-periods]
        test = serie.iloc[-periods:]
        test_dates = test.index

        results = {}
        errors = []

        if "prophet" in models:
            try:
                train_df = train.reset_index()
                train_df.columns = ["ds", "y"]
                m = Prophet(daily_seasonality=True)
                m.fit(train_df)
                future = m.make_future_dataframe(periods=periods, freq=freq)
                fcst = m.predict(future)
                yhat = fcst[["ds", "yhat"]].tail(periods)["yhat"]
                yhat.index = test_dates
                results["prophet"] = yhat
            except Exception as e:
                errors.append(f"prophet: {e}")

        if "arima" in models:
            try:
                train_arima = train.copy()
                try:
                    train_arima.index.freq = pd.tseries.frequencies.to_offset(freq)
                except ValueError:
                    pass  # índice no perfectamente regular; se deja que ARIMA infiera igual
                fit = ARIMA(train_arima, order=(2, 1, 2)).fit()
                fcst = fit.forecast(steps=periods)
                fcst.index = test_dates
                results["arima"] = fcst
            except Exception as e:
                errors.append(f"arima: {e}")

        if "sarima" in models:
            try:
                fcst = fit_sarima_forecast(train, freq, periods)
                fcst.index = test_dates
                results["sarima"] = fcst
            except Exception as e:
                errors.append(f"sarima: {e}")

        if "knn" in models:
            if len(train) <= n_lags:
                errors.append(f"knn: no hay suficientes datos de entrenamiento ({len(train)}) para {n_lags} rezagos.")
            else:
                try:
                    values = train.values
                    X, y = [], []
                    for i in range(n_lags, len(values)):
                        X.append(values[i - n_lags:i])
                        y.append(values[i])
                    X, y = np.array(X), np.array(y)

                    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
                    knn_model.fit(X, y)

                    last_window = list(values[-n_lags:])
                    preds = []
                    for _ in range(periods):
                        x_input = np.array(last_window[-n_lags:]).reshape(1, -1)
                        next_val = knn_model.predict(x_input)[0]
                        preds.append(next_val)
                        last_window.append(next_val)

                    results["knn"] = pd.Series(preds, index=test_dates)
                except Exception as e:
                    errors.append(f"knn: {e}")

        if not results:
            return f"Error: ningún modelo pudo entrenarse/predecir. Detalles: {errors}"

        # Calcular métricas de error por modelo
        metric_rows = []
        real = test.values
        for name, yhat in results.items():
            pred = yhat.values
            mae = mean_absolute_error(real, pred)
            rmse = np.sqrt(mean_squared_error(real, pred))
            mape = np.mean(np.abs((real - pred) / real)) * 100
            metric_rows.append({"Modelo": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE (%)": round(mape, 2)})
        metrics_df = pd.DataFrame(metric_rows)

        # Graficar con Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines",
                                  name="Real (histórico)", line=dict(color="#2a78d6", width=2)))

        color_map = {"prophet": "#1baf7a", "arima": "#eda100", "sarima": "#d6493a", "knn": "#4a3aa7"}
        for name in ["prophet", "arima", "sarima", "knn"]:
            if name in results:
                fig.add_trace(go.Scatter(x=test_dates, y=results[name].values, mode="lines+markers",
                                          name=f"Predicción {name}",
                                          line=dict(color=color_map[name], width=2, dash="dash")))

        fig.add_vline(x=train.index[-1], line_width=1, line_dash="dot", line_color="gray",
                      annotation_text="Inicio holdout", annotation_position="top")
        fig.update_layout(title=f"Comparación de modelos para {column} (holdout={horizon} días)",
                          xaxis_title="Fecha", yaxis_title=column, legend_title="Serie",
                          template="plotly_white")
        fig.show()

        summary = f"Comparación completada para '{column}' (horizon={horizon}).\n"
        summary += metrics_df.to_string(index=False)
        if errors:
            summary += "\n\nModelos no evaluados:\n" + "\n".join(errors)
        return summary

    except Exception as e:
        return f"Error durante la comparación de modelos: {e}"


tool_compare_models = {
  'type': 'function',
  'function': {
    'name': 'compare_models',
    'description': 'Compara predicciones de varios modelos (prophet, arima, sarima, knn) contra datos reales retenidos (holdout) y muestra un gráfico interactivo de Plotly con las métricas de error (MAE, RMSE, MAPE) de cada modelo. Usa esta herramienta cuando el usuario pida comparar modelos o evaluar qué tan buena es una predicción.',
    'parameters': {
      'type': 'object',
      'properties': {
        'column': {
          'type': 'string',
          'description': 'Nombre de la columna a predecir y comparar'
        },
        'horizon': {
          'type': 'integer',
          'description': 'Días retenidos como datos reales de prueba (holdout) y horizonte de predicción.'
        },
        'models': {
          'type': 'array',
          'items': {'type': 'string'},
          'description': 'Lista de modelos a comparar: subconjunto de ["prophet","arima","sarima","knn"]. Si no se especifica, se comparan prophet, arima y knn (sarima es más lento de ajustar y solo se incluye si se pide explícitamente).'
        },
        'n_neighbors': {
          'type': 'integer',
          'description': 'Número de vecinos a usar en KNN (por defecto 5), solo aplica si "knn" está en models.'
        },
        'n_lags': {
          'type': 'integer',
          'description': 'Número de rezagos usados como features en KNN (por defecto 7), solo aplica si "knn" está en models.'
        }
      },
      'required': ['column']
    }
  }
}


# =============================
# Diccionario de herramientas
# El LLM solo ve las herramientas en este diccionario
# =============================
dic_tools = {
    "load_csv": load_csv,
    "final_answer": final_answer,
    "code_exec": code_exec,
    "plot_data": plot_data,
    "predict_data": predict_data,
    "predict_knn": predict_knn,
    "compare_models": compare_models

}

# =============================
# Ejecutor de herramientas
# =============================

# Lee las tool calls que el LLM decidió invocar.
# Mapea y ejecuta la función Python correspondiente.

def use_tool(agent_res: dict, dic_tools: dict) -> dict:
    msg = agent_res["message"]
    res, t_name, t_inputs = "", "", ""

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            t_name = tool["function"]["name"]
            raw_args = tool["function"]["arguments"]

            # Parsear argumentos en formato JSON o dict
            try:
                t_inputs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                t_inputs = raw_args

            # 👇 Normalizar argumentos según la herramienta
            if t_name == "load_csv":
                t_inputs = normalize_csv_args(t_inputs)

            elif t_name == "plot_data":
                t_inputs = normalize_plot_args(t_inputs)

            elif t_name == "code_exec":
                if isinstance(t_inputs, dict):
                    code = t_inputs.get("code", "")
                    t_inputs = {"code": code}

            elif t_name == "predict_data":
                if isinstance(t_inputs, dict):
                    # Valores por defecto
                    t_inputs.setdefault("model", "prophet")
                    t_inputs.setdefault("horizon", 7)
                    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                        t_inputs["column"] = list(dtf.columns)[0]

            elif t_name == "predict_knn":
                if isinstance(t_inputs, dict):
                    # Valores por defecto
                    t_inputs.setdefault("horizon", 7)
                    t_inputs.setdefault("n_neighbors", 5)
                    t_inputs.setdefault("n_lags", 7)
                    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                        t_inputs["column"] = list(dtf.columns)[0]

            elif t_name == "compare_models":
                if isinstance(t_inputs, dict):
                    # Valores por defecto
                    t_inputs.setdefault("horizon", 7)
                    t_inputs.setdefault("n_neighbors", 5)
                    t_inputs.setdefault("n_lags", 7)
                    if "column" not in t_inputs or t_inputs["column"] not in dtf.columns:
                        t_inputs["column"] = list(dtf.columns)[0]

                    # Normalizar "models" (puede llegar como string, lista o None)
                    if "models" in t_inputs:
                        if isinstance(t_inputs["models"], str):
                            try:
                                t_inputs["models"] = json.loads(t_inputs["models"].replace("'", '"'))
                            except Exception:
                                t_inputs["models"] = [t_inputs["models"]]
                        elif t_inputs["models"] is None:
                            t_inputs.pop("models")

                    if "models" in t_inputs and isinstance(t_inputs["models"], list):
                        validos = [m for m in t_inputs["models"] if m in ("prophet", "arima", "sarima", "knn")]
                        if validos:
                            t_inputs["models"] = validos
                        else:
                            t_inputs.pop("models")

            elif t_name == "final_answer" and isinstance(t_inputs, dict) and "final_answer" in t_inputs:
                t_inputs = {"text": t_inputs["final_answer"]}

            # 🔧 Ejecutar la herramienta correspondiente
            if f := dic_tools.get(t_name):
                print(f"🔧 > {t_name} -> Inputs: {t_inputs}")
                try:
                    if isinstance(t_inputs, dict):
                        t_output = f(**t_inputs)
                    else:
                        t_output = f(t_inputs)
                except Exception as e:
                    cols = list(dtf.columns) if 'dtf' in globals() else 'No hay dataset cargado'
                    t_output = f"Error ejecutando {t_name}: {e}. Columnas disponibles: {cols}"

                # Mostrar resultado
                print(f"📊 Resultado: {t_output}")
                res = t_output
            else:
                print(f"🤬 > {t_name} -> NotFound")

    # Si el mensaje trae texto normal (sin herramientas)
    if msg.get("content", "") != "":
        res = msg["content"]
        t_name, t_inputs = "", ""

    return {"res": res, "tool_used": t_name, "inputs_used": t_inputs}










# =============================
# Bucle principal del agente
# =============================
# Hace el loop de llamadas con Ollama
# Pasa las tools (metadata) para que el LLM pueda decidir cuál usar.
#  Mantiene un historial de uso de herramientas

def run_agent(llm, messages, available_tools):
    tool_used, local_memory = '', ''
    used_compute = False

    while tool_used != 'final_answer':
        try:
            agent_res = ollama.chat(
                model=llm, 
                messages=messages, 
                #format="json", 
                tools=[v for v in available_tools.values()]
            )

            dic_res = use_tool(agent_res, dic_tools)
            res, tool_used, inputs_used = dic_res["res"], dic_res["tool_used"], dic_res["inputs_used"]

          
            if tool_used in ("code_exec", "plot_data"):
                used_compute = True

            user_query = messages[-1]["content"].lower()
            needs_compute = any(word in user_query for word in [
                "promedio", "media", "máximo", "mínimo", "suma", "resta",
                "gráfico", "grafica", "plot", "visualiza", "filtra",
                "porcentaje", "calcula", "valor", "estadística", "histograma", "error", 
            ])

            if tool_used == "final_answer" and needs_compute and not used_compute:
                print("⚠️ > El modelo intentó responder sin calcular. Reintentando...")
                messages.append({
                    "role": "user", 
                    "content": "Debes usar code_exec o plot_data antes de final_answer."
                })
                tool_used = ""
                continue

        except Exception as e:
            print("⚠️ >", e)
            res = f"Intenté usar {tool_used} pero falló. Intentaré otra cosa."
            messages.append({"role": "assistant", "content": res})

        if tool_used not in ['', 'final_answer']:
            # Agregar al historial de memoria
            local_memory += f"\nTool used: {tool_used}.\nInput used: {inputs_used}.\nOutput: {res}"
            messages.append({"role": "assistant", "content": f"Resultado: {res}"})
            available_tools.pop(tool_used, None)
            if len(available_tools) == 1:
                messages.append({"role": "user", "content": "ahora activa la herramienta final_answer."})

        if tool_used == '':
            break

    return res

prompt = '''
Eres un Analista de Datos experto en Python y pandas.
Tu tarea es responder cualquier consulta del usuario sobre el dataset `dtf`.

Reglas generales:
- El dataset siempre se llama `dtf` (ya cargado en memoria).
- Usa solo las columnas reales disponibles en dtf.columns.
- Nunca inventes datos, nombres de columnas ni valores numéricos.
- Siempre usa comillas dobles para los nombres de columnas: dtf["columna"].
- Nunca uses pd.read_csv() dentro de code_exec.

Reglas para cálculos:
- Usa la herramienta code_exec con una única línea completa y válida: print(...).
- Ejemplo válido: {"code": "print(dtf[\"MW\"].mean())"}
- Nunca generes código incompleto ni multilínea.
- Después de code_exec, usa final_answer para explicar el resultado en lenguaje natural.

Reglas para gráficos:
- Usa exclusivamente la herramienta plot_data.
- "columns" debe ser una lista JSON (ej: ["MW","MW_P"]).
- Si el usuario pide un día: start_date = end_date = "YYYY-MM-DD".
- "title" debe describir el gráfico claramente.
- Nunca uses matplotlib manualmente en code_exec.

Reglas para predicciones:
- Usa la herramienta predict_data para predicciones con "prophet", "arima" o "sarima".
- Usa "sarima" (dentro de predict_data) cuando el usuario pida explícitamente SARIMA o quiera capturar estacionalidad (ciclo diario/semanal) en la predicción de ARIMA.
- Usa la herramienta predict_knn cuando el usuario pida explícitamente KNN, K-Nearest Neighbors o "vecinos más cercanos".
- Siempre grafica los valores futuros junto con los históricos.
- Parámetros de predict_data: {"model": "prophet", "arima" o "sarima", "column": "MW", "horizon": 7}.
- Parámetros de predict_knn: {"column": "MW", "horizon": 7, "n_neighbors": 5, "n_lags": 7}.
- Nunca inventes valores de predicción; deben provenir de la ejecución real.
- Después de predecir, usa final_answer para explicar el resultado.

Reglas para comparación de modelos:
- Usa la herramienta compare_models cuando el usuario pida comparar modelos, evaluar qué tan buena es una predicción, o pida métricas de error (MAE, RMSE, MAPE) de una predicción.
- compare_models NO predice hacia el futuro: retiene los últimos "horizon" días como datos reales de prueba, entrena cada modelo con el resto, y compara la predicción contra esos datos reales retenidos.
- Parámetros: {"column": "MW", "horizon": 7, "models": ["prophet","arima","sarima","knn"]}.
- Si el usuario no especifica modelos, compara prophet, arima y knn. Solo incluye "sarima" si el usuario lo pide explícitamente (es más lento de ajustar).
- Siempre muestra el gráfico interactivo (Plotly) y el resumen de métricas devuelto por la herramienta.
- Después de comparar, usa final_answer para explicar cuál modelo tuvo mejor desempeño según las métricas.

Reglas para final_answer:
- Usa final_answer solo para texto descriptivo o interpretaciones.
- No inventes valores calculados; todos deben provenir de code_exec, plot_data o predict_data.

Flujo de decisión:
- Si requiere cálculo, estadística, gráfico o predicción → usa primero code_exec, plot_data o predict_data.
- Si es descriptivo o conceptual → usa final_answer.
- Si no hay datos disponibles → final_answer explicando la causa.


Ejemplos: 
- Usuario: "¿Qué columnas tiene el archivo?" → {"name":"final_answer","arguments":{"text":"Las columnas son ..."}} 
- Usuario: "¿Cuál es el promedio de MW?" → {"name":"code_exec","arguments":{"code":"print(dtf[\"MW\"].mean())"}} 
- Usuario: "Haz un gráfico del día 2024-09-06" → {"name":"plot_data","arguments":{"columns":["MW"],"start_date":"2024-09-06","end_date":"2024-09-06","title":"MW en 2024-09-06"}}
- Usuario: "¿Qué tan correlacionadas están MW y MW_P?" → {"name":"code_exec","arguments":{"code":"print(dtf[\"MW\"].corr(dtf[\"MW_P\"]))"}} 
- Usuario: "Haz una predicción de los próximos 7 días con Prophet para MW" → {"name":"predict_data","arguments":{"model":"prophet","column":"MW","horizon":7}}
- Usuario: "Predice MW con KNN para los próximos 5 días" → {"name":"predict_knn","arguments":{"column":"MW","horizon":5}}
- Usuario: "Compara los modelos Prophet, ARIMA y KNN para MW en los próximos 7 días" → {"name":"compare_models","arguments":{"column":"MW","horizon":7,"models":["prophet","arima","knn"]}}
- Usuario: "¿Qué tan buena es la predicción de Prophet para MW?" → {"name":"compare_models","arguments":{"column":"MW","horizon":7,"models":["prophet"]}}
- Usuario: "Predice MW con SARIMA para los próximos 3 días" → {"name":"predict_data","arguments":{"model":"sarima","column":"MW","horizon":3}}
- Usuario: "Compara ARIMA contra SARIMA para MW en los próximos 5 días" → {"name":"compare_models","arguments":{"column":"MW","horizon":5,"models":["arima","sarima"]}}
'''




messages = [{"role":"system", "content":prompt}]

# =============================
# Chat interactivo
# =============================
while True:
    q = input("🙂 > ")
    if q.lower() == "quit":
        break
    messages.append({"role": "user", "content": q})
    available_tools = {
        "load_csv": tool_load_csv,
        "final_answer": tool_final_answer,
        "code_exec": tool_code_exec,
        "plot_data": tool_plot_data,
        "predict_data": tool_predict_data,
        "predict_knn": tool_predict_knn,
        "compare_models": tool_compare_models
    }
    res = run_agent(llm, messages, available_tools)
    print("👽 >", res)
    messages.append({"role": "assistant", "content": res})
