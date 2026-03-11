#  Moteur de Recommandation E-commerce

> Système de recommandation personnalisé basé sur **PySpark ALS** et **TensorFlow Embeddings**, servi via une **API FastAPI** avec interface web interactive.

---

## Résultats Clés

| Modèle | RMSE | Amélioration |
|--------|------|-------------|
| ALS v1 (rank=50, LONG) | 2.5676 | baseline |
| ALS v2 (rank=100, FLOAT) | 2.5304 | -1.5% |
| **TensorFlow Embeddings** | **1.6519** | **-34.7% ** |

| Métrique API | Valeur | Objectif |
|-------------|--------|---------|
| Latence `/api/recommend` | **< 1ms** | < 50ms  |
| Stratégie Cold Start |  Implémentée | — |
| Utilisateurs dataset | 631 986 | — |
| Produits dataset | 112 565 | — |
| Interactions | 701 528 | — |

---

##  Architecture

```
Hugging Face          Databricks (PySpark)       Local PC
─────────────         ────────────────────       ────────
Amazon Reviews   →    ALS + TensorFlow      →    FastAPI
2023 dataset          MLflow tracking            Interface Web
                      Volumes storage            GitHub
```

**Pattern Batch/Serving :**
- **Offline** : Databricks calcule les recommandations en batch (top-100 par user)
- **Online** : FastAPI sert les résultats depuis la mémoire en < 1ms

---

##  Structure du Projet

```
recommendation-engine/
├── notebooks/
│   ├── 01_data_exploration/
│   │   └── explore_amazon_data.py       # Chargement Hugging Face + stats
│   ├── 02_feature_engineering/
│   │   └── build_features.py            # StringIndexer, split temporel, profils
│   ├── 03_models/
│   │   ├── als_model.py                 # ALS v1 et v2 + MLflow tracking
│   │   └── tensorflow_embeddings.py     # Embeddings 32 dims + Dense layers
│   └── 05_evaluation/
│       └── ab_testing.py                # Framework A/B test (z-test)
├── src/
│   └── api/
│       ├── main.py                      # Config FastAPI + chargement données
│       ├── routes/
│       │   └── recommend.py             # Endpoints REST
│       ├── static/
│       │   └── style.css                # Styles séparés de la logique
│       └── templates/
│           ├── home.html                # Dashboard d'accueil
│           ├── recommend.html           # Interface interactive
│           └── stats.html               # Statistiques des modèles
├── recommendations_api.json             # Export Databricks (non versionné)
├── requirements.txt
└── README.md
```


##  Lancer l'API

### Prérequis

```bash
pip install -r requirements.txt
```

```txt
# requirements.txt
fastapi
uvicorn
jinja2
python-multipart
```

### Démarrage

```bash
cd src/api
uvicorn main:app --reload --port 8000
```

### Pages disponibles

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Dashboard avec métriques |
| `http://localhost:8000/recommend` | Interface de test interactive |
| `http://localhost:8000/stats` | Statistiques des modèles |
| `http://localhost:8000/docs` | Documentation Swagger OpenAPI |
| `http://localhost:8000/api/recommend/{user_id}` | Endpoint JSON |
| `http://localhost:8000/api/health` | Statut de l'API |

---

##  Dataset

**Amazon Reviews 2023** — Catégorie *All Beauty*
- Source : [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (Hugging Face)
- Période : Novembre 2000 → Septembre 2023
- Densité matrice : 0.001% (ultra-creuse)
-  Biais : 61% des ratings sont 5 étoiles

---

##  Pipeline Databricks

### 1. Chargement des données

```python
from huggingface_hub import hf_hub_download
import shutil

# Reviews
reviews_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/All_Beauty.jsonl",
    repo_type="dataset"
)
shutil.copy(reviews_path, "/Volumes/workspace/default/ecommerce_data/")

# Metadata
meta_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw_meta_All_Beauty/full-00000-of-00001.parquet",
    repo_type="dataset"
)
shutil.copy(meta_path, "/Volumes/workspace/default/ecommerce_data/")
```

>  **Note** : `trust_remote_code=True` n'est pas supporté sur Databricks Community Edition. Utiliser `hf_hub_download` + `shutil.copy`.

### 2. Feature Engineering

```python
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

# Indexation obligatoire pour ALS (exige des entiers)
user_indexer = StringIndexer(inputCol="user_id",    outputCol="user_idx")
item_indexer = StringIndexer(inputCol="parent_asin", outputCol="item_idx")

# Conversion rating LONG → FLOAT (améliore la précision des gradients ALS)
df = df.withColumn("rating", col("rating").cast("float"))

# Split temporel (évite le data leakage)
split_date = "2021-01-01"
train = df.filter(col("date") < split_date)   # 71% — 500 645 interactions
test  = df.filter(col("date") >= split_date)  # 29% — 200 883 interactions
```

### 3. Modèle ALS

```python
from pyspark.ml.recommendation import ALS
import mlflow

with mlflow.start_run():
    als = ALS(
        rank=100,
        regParam=0.05,
        maxIter=15,
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    model = als.fit(train)

    predictions = model.transform(test)
    rmse = evaluator.evaluate(predictions)  # 2.5304

    mlflow.log_params({"rank": 100, "regParam": 0.05, "maxIter": 15})
    mlflow.log_metric("rmse", rmse)
```

>  **Contrainte Serverless** : `recommendForAllUsers()` est bloqué sur le cluster Serverless Unity Catalog. Solution : calcul manuel par produit scalaire des facteurs ALS.

### 4. Modèle TensorFlow

```python
import tensorflow as tf

# Architecture Embedding + Dense
user_input = tf.keras.Input(shape=(1,), name="user_idx")
item_input = tf.keras.Input(shape=(1,), name="item_idx")

user_emb = tf.keras.layers.Embedding(num_users, 32)(user_input)
item_emb = tf.keras.layers.Embedding(num_items, 32)(item_input)

user_flat = tf.keras.layers.Flatten()(user_emb)
item_flat = tf.keras.layers.Flatten()(item_emb)

concat = tf.keras.layers.Concatenate()([user_flat, item_flat])
dense1 = tf.keras.layers.Dense(64, activation="relu")(concat)
dense2 = tf.keras.layers.Dense(32, activation="relu")(dense1)
output = tf.keras.layers.Dense(1)(dense2)

model = tf.keras.Model([user_input, item_input], output)
model.compile(optimizer="adam", loss="mse")

# Entraînement sur 10% du dataset (contrainte cluster)
model.fit(X_train, y_train, epochs=5, batch_size=512)
# RMSE = 1.6519 → -34.7% vs ALS 
```

---

## 🔌 API — Exemple de Réponse

```bash
GET /api/recommend/AG73BVBKUOH22USSFJA5ZWL7AKXA?n=5
```

```json
{
  "user_id": "AG73BVBKUOH22USSFJA5ZWL7AKXA",
  "recommendations": [
    "B08BFH35VX",
    "B07ZS3DKL5",
    "B08GKHQ9P1",
    "B08G5YVHQP",
    "B01N5SHGS0"
  ],
  "strategy": "collaborative_filtering",
  "nb_recommendations": 5,
  "latency_ms": 0.045
}
```

**Cold Start** (utilisateur inconnu) :
```json
{
  "user_id": "NOUVEAU_USER_2024",
  "recommendations": ["B08BFH35VX", "B07ZS3DKL5", "..."],
  "strategy": "cold_start_popular",
  "nb_recommendations": 10,
  "latency_ms": 0.012
}
```

---

##  Stratégie Cold Start

| Cas | Condition | Stratégie | Badge UI |
|-----|-----------|-----------|----------|
| Utilisateur connu | `user_id` présent dans les recommandations | Collaborative Filtering (ALS) |  Bleu |
| Nouvel utilisateur | `user_id` absent | Produits populaires (fallback) | 🟠 Orange |

---

##  User IDs de test

```
# Utilisateur connu → Collaborative Filtering
AG73BVBKUOH22USSFJA5ZWL7AKXA

# Autre utilisateur connu
AHDVSLWHSORYGG3S5QZMVDFNOXUQ

# Nouvel utilisateur → Cold Start
NOUVEAU_USER_TEST_2024
```

---

##  Volumes Databricks

```
/Volumes/workspace/default/ecommerce_data/
├── reviews_beauty/          # Interactions brutes (Parquet)
├── metadata_beauty/         # Métadonnées produits (Parquet)
├── ratings_indexed/         # Données préparées pour ALS
├── user_profiles/           # Profils utilisateurs enrichis
├── product_profiles/        # Profils produits enrichis
├── user_mapping.json        # index → user_id
├── item_mapping.json        # index → product_id
├── als_model/               # Modèle ALS v1
├── als_model_v2/            # Modèle ALS v2 (meilleur)
├── tensorflow_model/        # Modèle TensorFlow
├── recommendations_als/     # Top-100 recommandations par user
└── recommendations_api.json # Export pour FastAPI
```

---

##  Stack Technique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Traitement distribué | PySpark (Databricks) | Scalabilité pour millions d'interactions |
| Collaborative Filtering | ALS (MLlib) | Factorisation matricielle |
| Deep Learning | TensorFlow 2.x | Embeddings + Dense layers |
| Tracking ML | MLflow (Databricks) | Comparaison et reproductibilité |
| Stockage | Volumes Databricks (Parquet) | Format colonnaire optimisé |
| API REST | FastAPI + Jinja2 | Serving < 1ms |
| Interface | HTML / CSS / JavaScript | Dashboard interactif |
| Versioning | GitHub | Historique et CI/CD |

---

##  Difficultés Techniques Résolues

| Problème | Cause | Solution |
|----------|-------|----------|
| Dataset inaccessible | Cluster Serverless bloque les connexions | `hf_hub_download` + `shutil.copy` |
| DBFS désactivé | Unity Catalog sur Community Edition | Volumes `/Volumes/workspace/...` |
| `recommendForAllUsers()` bloqué | Spark higher-order functions non supportées | Produit scalaire manuel des facteurs |
| RMSE élevé | 61% ratings = 5 étoiles + rating en LONG | Conversion LONG → FLOAT |
| CrossJoin timeout | 631K × 112K = 70 milliards de combinaisons | Batch de 100 users maximum |

---

##  Améliorations Futures

- [ ] **Redis Cache** — remplacer le JSON en mémoire pour un serving distribué
- [ ] **A/B Testing** — z-test pour valider statistiquement le +15% CTR
- [ ] **ANN (FAISS/ScaNN)** — remplacer le CrossJoin par Approximate Nearest Neighbors
- [ ] **Modèle hybride** — combiner ALS + TensorFlow + Content-Based
- [ ] **Retraining automatique** — Databricks Jobs pour mise à jour quotidienne
- [ ] **Docker** — containerisation API + Redis
- [ ] **Multi-catégories** — étendre à Electronics, Books, Home & Kitchen

---

##  Références

- [Amazon Reviews 2023 — McAuley Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- [PySpark ALS Documentation](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [MLflow Documentation](https://mlflow.org/docs/latest)
- [Databricks Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)

---

*Projet réalisé dans le cadre de la formation Big Data & Machine Learning *