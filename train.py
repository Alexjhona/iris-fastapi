"""Entrena el modelo con el dataset versionado y lo guarda en model/.

Antes de entrenar verifica que el CSV no haya cambiado (comparando su hash
SHA256 con data/DATA_VERSION.json). Así se garantiza que siempre se entrena
con los mismos datos y se obtiene el mismo modelo.

Uso:  python train.py
"""
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

SEED = 42  # semilla fija = resultados reproducibles
DATA_DIR = Path("data")
OUT = Path("model")
OUT.mkdir(exist_ok=True)

# 1. Cargar y verificar la versión de los datos
ficha = json.loads((DATA_DIR / "DATA_VERSION.json").read_text(encoding="utf-8"))
csv_path = DATA_DIR / ficha["archivo"]
sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
if sha256 != ficha["sha256"]:
    raise SystemExit(
        f"ERROR: {csv_path} no coincide con la versión {ficha['version']} registrada.\n"
        "Los datos fueron modificados. Restaura el archivo o crea una nueva versión."
    )
print(f"Datos verificados: {csv_path} (versión {ficha['version']}, SHA256 OK)")

df = pd.read_csv(csv_path)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X, y = df[FEATURES], df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 2. Entrenar
model = RandomForestClassifier(n_estimators=100, random_state=SEED)
model.fit(X_train.values, y_train)

# 3. Evaluar
y_pred = model.predict(X_test.values)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {acc:.4f}")
print(classification_report(y_test, y_pred))

# 4. Guardar modelo + metadatos (los usa la API)
joblib.dump(model, OUT / "model.joblib")
meta = {
    "modelo": "RandomForestClassifier",
    "hiperparametros": {"n_estimators": 100, "random_state": SEED},
    "dataset": f"Iris (150 muestras, 4 variables, 3 clases)",
    "version_datos": ficha["version"],
    "sha256_datos": ficha["sha256"],
    "clases": [str(c) for c in model.classes_],
    "variables": FEATURES,
    "accuracy_test": round(acc, 4),
    "entrenado_en": datetime.now().isoformat(timespec="seconds"),
    "python": platform.python_version(),
    "scikit_learn": sklearn.__version__,
}
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
print("Modelo guardado en model/model.joblib")
