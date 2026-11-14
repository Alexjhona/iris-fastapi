# API de predicción de flores Iris con FastAPI

Despliegue mínimo de un modelo de Machine Learning (Random Forest) como API REST con **FastAPI**, documentada con **Swagger UI** y **Scalar**.

- **Modelo:** RandomForestClassifier (100 árboles, semilla 42)
- **Datos:** Iris v1 (150 filas, 4 variables, 3 especies)
- **Accuracy en test:** 0.90

---

## 1. Requisitos

| Herramienta | Versión |
|---|---|
| Python | 3.13 (también probado en 3.14) |
| pip | 24 o superior |
| Git | cualquier versión reciente |

Las librerías están **fijadas con versión exacta** en [`requirements.txt`](requirements.txt) para que cualquiera instale exactamente lo mismo:

| Librería | Versión | Para qué sirve |
|---|---|---|
| fastapi | 0.162.0 | Framework de la API |
| uvicorn | 0.52.4 | Servidor que ejecuta la API |
| pydantic | 2.14.3 | Validación de datos de entrada |
| scikit-learn | 1.8.0 | Modelo Random Forest |
| numpy / scipy | 2.4.4 / 1.17.1 | Cálculo numérico (dependencias de scikit-learn) |
| pandas | 3.1.1 | Lectura del dataset CSV |
| joblib | 1.5.3 | Guardar y cargar el modelo |
| scalar-fastapi | 1.9.1 | Documentación con Scalar |

## 2. Cómo reproducir el proyecto (paso a paso)

```bash
# 1. Clonar el repositorio
git clone https://github.com/Alexjhona/iris-fastapi.git
cd iris-fastapi

# 2. Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / Mac

# 3. Instalar las dependencias exactas
pip install -r requirements.txt

# 4. Entrenar el modelo (verifica la versión de los datos)
python train.py

# 5. Levantar la API
python -m uvicorn main:app --reload
```

Abrir en el navegador:
- Swagger: http://127.0.0.1:8000/docs
- Scalar: http://127.0.0.1:8000/scalar

**Resultado esperado de `python train.py`:** `Datos verificados ... SHA256 OK` y `Accuracy en test: 0.9000`. Si sale otro valor, algo en el entorno es distinto.

## 3. Versionado de datos

El dataset está guardado dentro del repositorio y tiene una versión registrada:

```
data/
├── iris_v1.csv          ← datos usados para entrenar (versión v1)
└── DATA_VERSION.json    ← ficha: versión, origen, nº de filas y hash SHA256
```

- **`prepare_data.py`** creó `iris_v1.csv` a partir de `sklearn.datasets.load_iris` y calculó su **hash SHA256** (una "huella digital" del archivo).
- **`train.py`** vuelve a calcular el hash antes de entrenar. Si el CSV fue modificado aunque sea en un carácter, **se detiene con un error**. Así se garantiza que siempre se entrena con los mismos datos.
- El modelo guarda en `model/metadata.json` qué versión de datos usó (`version_datos` y `sha256_datos`). Se puede consultar en el endpoint `GET /model-info`.
- `.gitattributes` evita que Git cambie los saltos de línea del CSV en Windows (eso alteraría el hash).

**¿Cómo crear una nueva versión de datos?** Guardar el nuevo archivo como `data/iris_v2.csv`, actualizar `DATA_VERSION.json` (versión, archivo, filas, nuevo SHA256) y hacer un commit con un mensaje claro, por ejemplo `datos: iris v2`. El historial de Git queda como registro de todas las versiones.

## 4. Reproducibilidad del modelo

- Semilla fija (`random_state=42`) en la división train/test y en el Random Forest: entrenar dos veces da **el mismo modelo**.
- Versiones exactas de librerías en `requirements.txt` y de Python en `.python-version`.
- El modelo (`model/`) **no se sube a GitHub**: se regenera con `python train.py`. Así el repositorio solo guarda código + datos, y el modelo siempre se puede reconstruir.

## 5. Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Mensaje de bienvenida |
| GET | `/health` | Verifica que la API está activa |
| GET | `/model-info` | Modelo, versión de datos, accuracy y versiones usadas |
| POST | `/predict` | Recibe 4 medidas y devuelve la especie y sus probabilidades |
| GET | `/docs` | Documentación Swagger UI |
| GET | `/scalar` | Documentación Scalar |

Ejemplo de petición a `POST /predict`:

```json
{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
```

Respuesta:

```json
{"clase": "virginica", "probabilidades": {"setosa": 0.0, "versicolor": 0.0, "virginica": 1.0}}
```

## 6. Despliegue en la nube (opcional)

El proyecto se ejecuta de forma local (sección 2). Si más adelante se quiere publicar en internet, ya está preparado para Render.

El archivo [`render.yaml`](render.yaml) tiene la configuración. En Render:

- **Build command:** `pip install -r requirements.txt && python train.py`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

También se incluye un `Dockerfile` para desplegar en cualquier plataforma con Docker.

> En el plan gratuito de Render, el servicio "se duerme" tras 15 minutos sin uso. La primera visita después tarda unos 30–60 segundos en responder.

## 7. Estructura del proyecto

```
.
├── data/
│   ├── iris_v1.csv          # dataset versionado
│   └── DATA_VERSION.json    # ficha de la versión de datos (hash SHA256)
├── prepare_data.py          # crea la versión de datos (ya ejecutado)
├── train.py                 # verifica datos, entrena y guarda el modelo
├── main.py                  # API FastAPI (Swagger + Scalar)
├── requirements.txt         # dependencias con versión exacta
├── .python-version          # versión de Python
├── render.yaml              # configuración de despliegue en Render
├── Dockerfile               # despliegue alternativo con Docker
├── .gitignore               # archivos que no se suben (venv, model/, cache)
└── .gitattributes           # protege el CSV de cambios de salto de línea
```
