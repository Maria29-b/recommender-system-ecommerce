from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os

# ── Configuration ─────────────────────────────────────
app = FastAPI(
    title=" Moteur de Recommandation E-commerce",
    description="""
##  Système de recommandation personnalisé

Ce moteur de recommandation utilise **deux algorithmes** :
-  **ALS** (Alternating Least Squares) — Collaborative Filtering
-  **TensorFlow Embeddings** — Deep Learning

###  Performance
| Modèle | RMSE |
|--------|------|
| ALS | 2.53 |
| TensorFlow | 1.65  |

###  Dataset
-  **631,986** utilisateurs
-  **112,565** produits
-  **701,528** interactions
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# ── Fichiers statiques et templates ───────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Chargement des données ────────────────────────────
RECS_PATH = "recommendations_api.json"
if os.path.exists(RECS_PATH):
    with open(RECS_PATH) as f:
        RECOMMENDATIONS = json.load(f)
else:
    RECOMMENDATIONS = {}

POPULAR_PRODUCTS = [
    "B08BFH35VX", "B07ZS3DKL5", "B08GKHQ9P1",
    "B08G5YVHQP", "B01N5SHGS0", "B00O2FGBJS",
    "B0837K9W6P", "B01B5KTY3W", "B0859CYSTM",
    "B00KRMOJ4Y"
]

STATS = {
    "dataset": {
        "interactions" : 701528,
        "users"        : 631986,
        "products"     : 112565,
        "density"      : "0.001%"
    },
    "models": {
        "ALS"        : {"rmse": 2.5304, "precision_at_10": 0.004},
        "TensorFlow" : {"rmse": 1.6519, "embedding_dim": 32}
    }
}

# ── Importer les routes ───────────────────────────────
from routes.recommend import router
app.include_router(router)