from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import time
import sys
sys.path.append("..")

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

# ── Models ────────────────────────────────────────────
class RecommendationResponse(BaseModel):
    user_id            : str
    recommendations    : list
    strategy           : str
    nb_recommendations : int
    latency_ms         : float

# ── Pages HTML ────────────────────────────────────────
@router.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False
)
def home(request: Request):
    from main import RECOMMENDATIONS
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "nb_users": len(RECOMMENDATIONS)}
    )

@router.get(
    "/recommend",
    response_class=HTMLResponse,
    include_in_schema=False
)
def recommend_page(request: Request):
    return templates.TemplateResponse(
        "recommend.html",
        {"request": request}
    )

@router.get(
    "/stats",
    response_class=HTMLResponse,
    include_in_schema=False
)
def stats_page(request: Request):
    from main import STATS
    return templates.TemplateResponse(
        "stats.html",
        {"request": request, "stats": STATS}
    )

# ── API Endpoints ─────────────────────────────────────
@router.get(
    "/api/recommend/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommandations"],
    summary="Recommandations personnalisées"
)
def recommend_api(user_id: str, n: int = 10):
    from main import RECOMMENDATIONS, POPULAR_PRODUCTS

    #  time.perf_counter() = plus précis
    start = time.perf_counter()

    if user_id in RECOMMENDATIONS:
        recs     = RECOMMENDATIONS[user_id][:n]
        strategy = "collaborative_filtering"
    else:
        recs     = POPULAR_PRODUCTS[:n]
        strategy = "cold_start_popular"

    latency = (time.perf_counter() - start) * 1000

    #  minimum 0.01ms pour éviter 0ms
    latency = max(latency, 0.01)

    return RecommendationResponse(
        user_id            = user_id,
        recommendations    = recs,
        strategy           = strategy,
        nb_recommendations = len(recs),
        latency_ms         = round(latency, 3)  # 3 décimales
    )

@router.get(
    "/api/health",
    tags=["Monitoring"],
    summary="Statut de l'API"
)
def health():
    from main import RECOMMENDATIONS
    return {
        "status"       : "✅ healthy",
        "users_loaded" : len(RECOMMENDATIONS),
        "version"      : "1.0.0"
    }