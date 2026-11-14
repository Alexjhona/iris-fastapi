"""API mínima con FastAPI para servir el modelo predictivo.

Ejecutar:  uvicorn main:app --reload
Docs:      http://127.0.0.1:8000/docs    (Swagger)
           http://127.0.0.1:8000/scalar  (Scalar)
"""
import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference

MODEL_DIR = Path(__file__).parent / "model"
model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "metadata.json").read_text())

app = FastAPI(
    title="API de Predicción - Flores Iris",
    description="Despliegue mínimo de un modelo RandomForest con FastAPI.",
    version="1.0.0",
)


# ---------- Esquemas (aparecen documentados en Swagger/Scalar) ----------
class Flor(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Largo del sépalo (cm)", examples=[5.1])
    sepal_width: float = Field(..., gt=0, description="Ancho del sépalo (cm)", examples=[3.5])
    petal_length: float = Field(..., gt=0, description="Largo del pétalo (cm)", examples=[1.4])
    petal_width: float = Field(..., gt=0, description="Ancho del pétalo (cm)", examples=[0.2])


class Prediccion(BaseModel):
    clase: str
    probabilidades: dict[str, float]


# ---------- Endpoints ----------
@app.get("/", tags=["General"])
def inicio():
    return {"mensaje": "API funcionando. Ve a /docs (Swagger) o /scalar (Scalar)."}


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok"}


@app.get("/model-info", tags=["Modelo"])
def model_info():
    """Información y métricas del modelo cargado."""
    return meta


@app.post("/predict", response_model=Prediccion, tags=["Modelo"])
def predict(flor: Flor):
    """Predice la especie de una flor Iris a partir de sus medidas."""
    x = [[flor.sepal_length, flor.sepal_width, flor.petal_length, flor.petal_width]]
    clase = str(model.predict(x)[0])
    probas = model.predict_proba(x)[0]
    return Prediccion(
        clase=clase,
        probabilidades={str(c): round(float(p), 4) for c, p in zip(model.classes_, probas)},
    )


# ---------- Documentación con Scalar ----------
@app.get("/scalar", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)
