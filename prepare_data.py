"""Genera el dataset versionado data/iris_v1.csv y su ficha data/DATA_VERSION.json.

Solo se ejecuta UNA vez para crear la versión de los datos. El CSV queda
guardado en el repositorio, así que entrenar no depende de descargas externas.

Uso:  python prepare_data.py
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
VERSION = "v1"
CSV_PATH = DATA_DIR / f"iris_{VERSION}.csv"

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
})
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")
df.to_csv(CSV_PATH, index=False, lineterminator="\n")

sha256 = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
ficha = {
    "version": VERSION,
    "archivo": CSV_PATH.name,
    "origen": "sklearn.datasets.load_iris (UCI Iris, Fisher 1936)",
    "filas": len(df),
    "columnas": list(df.columns),
    "sha256": sha256,
    "fecha_creacion": date.today().isoformat(),
}
(DATA_DIR / "DATA_VERSION.json").write_text(json.dumps(ficha, indent=2, ensure_ascii=False) + "\n")
print(f"Dataset {CSV_PATH} creado ({len(df)} filas). SHA256: {sha256}")
