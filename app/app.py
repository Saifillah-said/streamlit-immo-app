"""
🏠 Application Immobilier — 3 pages
Page 1 : Upload CSV + Exploration
Page 2 : Entraînement du modèle + Performances
Page 3 : Interface de prédiction
"""

import io
import joblib
import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

from data import load_data, impute, prepare_data
from training import get_model_defs, train_and_evaluate
from test import predict_from_input

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model     import LinearRegression
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline         import Pipeline

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("immo_predict")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

logger.info("Démarrage de l'application")

# ── Authentification basique (utilisateurs définis par variables d'environnement)
def _get_auth_users() -> dict:
    """Retourne un dict username->password depuis les variables d'environnement."""
    return {
        os.environ.get("IMMO_USER", "admin"): os.environ.get("IMMO_PASS", "admin")
    }


def _check_auth(user: str, password: str) -> bool:
    users = _get_auth_users()
    valid = users.get(user) == password
    logger.info("Tentative de connexion utilisateur=%s succès=%s", user, valid)
    return valid


def _require_login() -> None:
    """Affiche le formulaire de connexion si l'utilisateur n'est pas authentifié."""
    if st.session_state.get("authenticated", False):
        return

    st.markdown("## 🔐 Connexion")
    user = st.text_input("Utilisateur", key="auth_user")
    password = st.text_input("Mot de passe", type="password", key="auth_pass")
    if st.button("Se connecter", key="auth_submit"):
        if _check_auth(user, password):
            logger.info("Authentification réussie pour %s", user)
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Identifiants invalides.")
    st.stop()


_require_login()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONFIG PAGE                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="🏠 ImmoPredict",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CSS GLOBAL                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px; padding: 2.2rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(229,160,64,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'DM Serif Display', serif; font-size: 2.3rem;
    color: #f5e6c8; margin: 0; letter-spacing: -0.02em; line-height: 1.1;
}
.hero p  { color: #a8b4c8; font-size: 0.95rem; margin-top: 0.4rem; font-weight: 300; }
.hero .badge {
    display: inline-block; background: rgba(229,160,64,0.2);
    border: 1px solid rgba(229,160,64,0.4); color: #e5a040;
    border-radius: 20px; padding: 0.18rem 0.75rem; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.7rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #fff; border: 1px solid #e8ecf0; border-radius: 12px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.metric-card .label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: #8899aa; margin-bottom: 0.25rem;
}
.metric-card .value {
    font-family: 'DM Serif Display', serif; font-size: 1.7rem; color: #1a1a2e; line-height: 1;
}
.metric-card .sub { font-size: 0.75rem; color: #a0aab4; margin-top: 0.2rem; }

/* ── Section title ── */
.section-title {
    font-family: 'DM Serif Display', serif; font-size: 1.35rem; color: #1a1a2e;
    margin: 1.4rem 0 0.7rem; padding-bottom: 0.35rem;
    border-bottom: 2px solid #e5a040; display: inline-block;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]               { background: #1a1a2e !important; }
[data-testid="stSidebar"] *             { color: #d4dce8 !important; }
[data-testid="stSidebar"] label         { font-weight: 500 !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3            { font-family: 'DM Serif Display', serif !important; color: #f5e6c8 !important; }

/* ── Nav buttons in sidebar ── */
.nav-btn {
    display: block; width: 100%; text-align: left;
    background: transparent; border: 1px solid rgba(229,160,64,0.25);
    border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.4rem;
    color: #d4dce8 !important; font-size: 0.9rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
}
.nav-btn:hover, .nav-btn.active {
    background: rgba(229,160,64,0.15) !important; border-color: #e5a040 !important; color: #f5e6c8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f4f6f9; padding: 4px; border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 0.38rem 1.1rem;
    font-weight: 500; font-size: 0.87rem; color: #556677;
}
.stTabs [aria-selected="true"] { background: #1a1a2e !important; color: #f5e6c8 !important; }

/* ── Download button ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #e5a040, #d4892a) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: opacity 0.2s !important;
}
.stDownloadButton button:hover { opacity: 0.88 !important; }

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #0f3460, #1a1a2e) !important;
    color: #f5e6c8 !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Info / result box ── */
.info-box {
    background: #f0f7ff; border-left: 4px solid #0f3460;
    border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
    font-size: 0.87rem; color: #334455; margin: 0.7rem 0;
}
.result-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border-radius: 14px; padding: 1.8rem 2rem; text-align: center; margin: 1rem 0;
}
.result-box .price {
    font-family: 'DM Serif Display', serif; font-size: 3rem;
    color: #e5a040; line-height: 1;
}
.result-box .label { color: #a8b4c8; font-size: 0.9rem; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PALETTES & STYLE MATPLOTLIB                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
DARK_BLUE = "#0f3460"
GOLD      = "#e5a040"
TEAL      = "#1a8fa0"
CORAL     = "#e06060"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.facecolor":     "#fafbfc",
    "figure.facecolor":   "white",
    "axes.grid":          True,
    "grid.color":         "#e8ecf0",
    "grid.linewidth":     0.7,
})

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
if "page"          not in st.session_state: st.session_state.page = "Page 1"
if "df_raw"        not in st.session_state: st.session_state.df_raw = None
if "trained_model" not in st.session_state: st.session_state.trained_model = None
if "model_features" not in st.session_state: st.session_state.model_features = None
if "label_encoders" not in st.session_state: st.session_state.label_encoders = {}
if "model_name"    not in st.session_state: st.session_state.model_name = None
if "metrics"       not in st.session_state: st.session_state.metrics = None

def metric_card(col, label, value, sub=""):
    col.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — NAVIGATION                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("## 🏠 ImmoPredict")
    st.markdown("---")
    st.markdown("### Navigation")

    pages = {
        "Page 1": "📂  Upload & Exploration",
        "Page 2": "🤖  Entraînement & Performances",
        "Page 3": "🔮  Prédiction",
    }
    for key, label in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key

    st.markdown("---")
    # Statuts
    has_data  = st.session_state.df_raw is not None
    has_model = st.session_state.trained_model is not None
    st.markdown(
        f"{'✅' if has_data  else '⭕'} Données chargées\n\n"
        f"{'✅' if has_model else '⭕'} Modèle entraîné",
    )

    if st.session_state.get("authenticated", False):
        if st.button("🔓 Se déconnecter", key="logout"):
            st.session_state.authenticated = False
            st.experimental_rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#6688aa;'>Ames Housing Dataset<br>1 460 obs · 80 variables</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — UPLOAD & EXPLORATION                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
if page == "Page 1":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 1 · Dataset</div>
      <h1>Upload &amp; Exploration</h1>
      <p>Chargez votre dataset, appliquez des filtres et explorez les distributions.</p>
    </div>""", unsafe_allow_html=True)

    # ── Upload ─────────────────────────────────────────────────────
    section("Chargement du dataset")
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "📂 Charger train.csv",
            type="csv",
            help="Format Ames Housing recommandé. Si vide, le fichier train.csv local est utilisé.",
        )
    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>Format attendu</b><br>
        CSV avec colonnes Ames Housing :<br>
        SalePrice, GrLivArea, Neighborhood…
        </div>""", unsafe_allow_html=True)

    if uploaded:
        st.session_state.df_raw = load_data(uploaded)
    elif st.session_state.df_raw is None:
        try:
            st.session_state.df_raw = load_data("train.csv")
        except FileNotFoundError:
            st.error("⚠️ Aucun fichier trouvé. Chargez votre CSV via le widget ci-dessus.", icon="🚨")
            st.stop()

    df_raw = st.session_state.df_raw

    # ── Analyse des valeurs manquantes ─────────────────────────────
    section("Analyse des valeurs manquantes")
    missing = df_raw.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("✅ Aucune valeur manquante détectée.")
    else:
        st.markdown(f"Colonnes avec valeurs manquantes : **{len(missing)} / {df_raw.shape[1]}**")
        fig, ax = plt.subplots(figsize=(10, min(6, len(missing)*0.25 + 1)))
        missing.head(20).plot(kind='barh', ax=ax, color='tomato', edgecolor='white')
        ax.set_xlabel("% manquant")
        ax.set_title("Taux de valeurs manquantes (top 20)")
        st.pyplot(fig, use_container_width=True); plt.close()

    # ── Filtres ────────────────────────────────────────────────────
    section("Filtres interactifs")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        p_min, p_max = int(df_raw["SalePrice"].min()), int(df_raw["SalePrice"].max())
        price_range = st.slider("💰 Plage de prix", p_min, p_max, (p_min, p_max),
                                step=5000, format="%d$",
                                help="Filtrer par prix de vente.")
    with fc2:
        neighborhoods = sorted(df_raw["Neighborhood"].unique())
        sel_neigh = st.multiselect("📍 Quartier(s)", neighborhoods, default=neighborhoods,
                                   help="Sélectionner un ou plusieurs quartiers.")
    with fc3:
        q_min, q_max = int(df_raw["OverallQual"].min()), int(df_raw["OverallQual"].max())
        qual_range = st.slider("⭐ Qualité générale", q_min, q_max, (q_min, q_max),
                               help="Note de qualité globale (1=mauvais, 10=excellent).")

    fc4, fc5 = st.columns(2)
    with fc4:
        s_min, s_max = int(df_raw["GrLivArea"].min()), int(df_raw["GrLivArea"].max())
        surf_range = st.slider("📐 Surface habitable (pi²)", s_min, s_max, (s_min, s_max),
                               step=100, help="GrLivArea — surface au-dessus du sol.")
    with fc5:
        zones = sorted(df_raw["MSZoning"].unique())
        sel_zones = st.multiselect("🗺️ Zone (MSZoning)", zones, default=zones,
                                   help="Classe de zonage de la propriété.")

    df = df_raw[
        df_raw["SalePrice"].between(*price_range) &
        df_raw["Neighborhood"].isin(sel_neigh) &
        df_raw["OverallQual"].between(*qual_range) &
        df_raw["GrLivArea"].between(*surf_range) &
        df_raw["MSZoning"].isin(sel_zones)
    ].copy()

    if len(df) == 0:
        st.error("Aucune donnée ne correspond aux filtres. Élargissez votre sélection.", icon="🚫")
        st.stop()

    # ── KPIs ───────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    metric_card(c1, "Biens sélectionnés", f"{len(df):,}", f"{len(df)/len(df_raw)*100:.1f}% du total")
    metric_card(c2, "Prix moyen",  f"{df['SalePrice'].mean()/1000:.0f}k$", f"médiane {df['SalePrice'].median()/1000:.0f}k$")
    metric_card(c3, "Surface moy.", f"{df['GrLivArea'].mean():.0f}", "pi²")
    metric_card(c4, "Quartiers",   str(df["Neighborhood"].nunique()), "sélectionnés")
    metric_card(c5, "Qualité moy.", f"{df['OverallQual'].mean():.1f}", "/10")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Onglets visualisation ──────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions", "🔵 Surface vs Prix",
        "📦 Boxplot Quartiers", "🔥 Corrélations", "📋 Données brutes",
    ])

    # TAB 1 — Histogramme
    with tab1:
        section("Distribution des variables")
        ca, cb = st.columns([2,1])
        with ca:
            col_var = st.selectbox("Variable", ["SalePrice","GrLivArea","SurfaceTotale",
                                                  "AgeLogement","OverallQual","GarageArea"],
                                   help="Variable à visualiser.")
        with cb:
            col_log = st.checkbox("Échelle log", value=(col_var=="SalePrice"),
                                  help="Transformation log1p pour les variables asymétriques.")

        data_plot = np.log1p(df[col_var]) if col_log else df[col_var]
        lx = f"log1p({col_var})" if col_log else col_var

        fig, axes = plt.subplots(1,2,figsize=(14,4.5))
        axes[0].hist(data_plot.dropna(), bins=40, color=DARK_BLUE, edgecolor="white", alpha=0.9)
        axes[0].axvline(data_plot.mean(),   color=GOLD,  ls="--", lw=1.6, label=f"Moy {data_plot.mean():.2f}")
        axes[0].axvline(data_plot.median(), color=CORAL, ls=":",  lw=1.6, label=f"Méd {data_plot.median():.2f}")
        axes[0].set_xlabel(lx); axes[0].set_ylabel("Effectif")
        axes[0].set_title(f"Histogramme — {col_var}", fontweight="600"); axes[0].legend(fontsize=9)

        sns.kdeplot(data_plot.dropna(), ax=axes[1], fill=True, color=TEAL, alpha=0.3, linewidth=2)
        axes[1].set_xlabel(lx); axes[1].set_ylabel("Densité")
        axes[1].set_title(f"Densité KDE — {col_var}", fontweight="600")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        info(f"📌 <b>{col_var}</b> — moy: <b>{df[col_var].mean():,.1f}</b> · "
             f"méd: <b>{df[col_var].median():,.1f}</b> · "
             f"std: <b>{df[col_var].std():,.1f}</b> · skew: <b>{df[col_var].skew():.2f}</b>")

    # TAB 2 — Scatter
    with tab2:
        section("Surface habitable vs Prix de vente")
        col_color = st.selectbox("Colorer par", ["OverallQual","Neighborhood","AgeLogement"],
                                 help="Variable de couleur des points.")
        fig, ax = plt.subplots(figsize=(12,6))
        cats = df[col_color].astype("category").cat.codes if df[col_color].dtype=="object" or df[col_color].nunique()<=15 else df[col_color]
        sc = ax.scatter(df["GrLivArea"], df["SalePrice"], c=cats, cmap="viridis", alpha=0.5, s=18, linewidths=0)
        z = np.polyfit(df["GrLivArea"].dropna(), df["SalePrice"].dropna(), 1)
        xs = np.linspace(df["GrLivArea"].min(), df["GrLivArea"].max(), 200)
        ax.plot(xs, np.poly1d(z)(xs), color=CORAL, lw=2, ls="--", label="Tendance")
        plt.colorbar(sc, ax=ax, label=col_color, pad=0.01)
        ax.set_xlabel("GrLivArea (pi²)", fontsize=12); ax.set_ylabel("Prix ($)", fontsize=12)
        ax.set_title("Scatter — Surface vs Prix", fontweight="600", pad=14)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
        r = df["GrLivArea"].corr(df["SalePrice"])
        ax.text(0.03,0.95,f"r = {r:.3f}", transform=ax.transAxes, fontsize=11, fontweight="600",
                color=DARK_BLUE, va="top", bbox=dict(boxstyle="round,pad=0.3",facecolor="#f0f7ff",edgecolor="#c0d0e0"))
        ax.legend(fontsize=9); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # TAB 3 — Boxplot
    with tab3:
        section("Distribution par quartier")
        cb1, cb2 = st.columns(2)
        with cb1:
            box_var = st.selectbox("Variable Y", ["SalePrice","GrLivArea","OverallQual","AgeLogement"])
        with cb2:
            sort_by = st.radio("Trier par", ["Médiane ↓","Médiane ↑","Alpha"], horizontal=True)

        meds = df.groupby("Neighborhood")[box_var].median()
        order = (meds.sort_values(ascending=False).index.tolist() if sort_by=="Médiane ↓"
                 else meds.sort_values().index.tolist() if sort_by=="Médiane ↑"
                 else sorted(df["Neighborhood"].unique()))

        fig, ax = plt.subplots(figsize=(14,6))
        bp = ax.boxplot([df[df["Neighborhood"]==nb][box_var].dropna().values for nb in order],
                        labels=order, patch_artist=True, widths=0.55,
                        medianprops=dict(color=GOLD,lw=2),
                        whiskerprops=dict(color="#aabbcc",lw=1.2),
                        capprops=dict(color="#aabbcc",lw=1.2),
                        flierprops=dict(marker="o",markerfacecolor=CORAL,alpha=0.4,markersize=3,ls="none"))
        for patch, color in zip(bp["boxes"], sns.color_palette("muted", n_colors=len(order))):
            patch.set_facecolor((*color[:3], 0.75))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylabel(box_var, fontsize=12)
        ax.set_title(f"Box plot — {box_var} par Neighborhood", fontweight="600", pad=14)
        if box_var == "SalePrice":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        summary = df.groupby("Neighborhood")[box_var].agg(
            N="count", Moyenne="mean", Médiane="median", Écart_type="std"
        ).round(0).sort_values("Médiane", ascending=False)
        st.dataframe(summary.style.background_gradient(cmap="Blues", subset=["Médiane"]),
                     use_container_width=True)

    # TAB 4 — Corrélations
    with tab4:
        section("Matrice de corrélation")
        num_def = [c for c in ["SalePrice","GrLivArea","SurfaceTotale","OverallQual",
                                "AgeLogement","GarageArea","TotalBsmtSF","NbSallesDeBain",
                                "LotArea","TotRmsAbvGrd","Fireplaces"] if c in df.columns]
        sel_corr = st.multiselect("Variables", options=[c for c in df.select_dtypes("number").columns if c!="Id"],
                                  default=num_def, help="Variables pour la matrice de Pearson.")
        if len(sel_corr) < 2:
            st.info("Sélectionnez au moins 2 variables.", icon="ℹ️")
        else:
            corr_m = df[sel_corr].corr()
            fig, ax = plt.subplots(figsize=(max(8,len(sel_corr)*0.75), max(6,len(sel_corr)*0.65)))
            sns.heatmap(corr_m, mask=np.triu(np.ones_like(corr_m,dtype=bool),k=1),
                        ax=ax, annot=True, fmt=".2f", cmap="RdYlBu_r", vmin=-1, vmax=1,
                        square=True, linewidths=0.4, linecolor="white",
                        annot_kws={"size":8}, cbar_kws={"shrink":0.8})
            ax.set_title("Matrice de corrélation", fontweight="600", pad=14)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0,  labelsize=9)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # TAB 5 — Données brutes
    with tab5:
        section("Données filtrées")
        ca, cb = st.columns([3,1])
        with ca:
            search = st.text_input("🔍 Rechercher", placeholder="CollgCr, RL, 2008…",
                                   help="Recherche textuelle sur toutes les colonnes.")
        with cb:
            n_rows = st.selectbox("Lignes", [25,50,100,200,"Toutes"], index=1)

        df_disp = df.copy()
        if search:
            df_disp = df_disp[df_disp.astype(str).apply(
                lambda c: c.str.contains(search, case=False, na=False)).any(axis=1)]
        if n_rows != "Toutes":
            df_disp = df_disp.head(int(n_rows))

        prio = [c for c in ["Id","Neighborhood","MSZoning","SalePrice","GrLivArea",
                              "SurfaceTotale","OverallQual","AgeLogement"] if c in df_disp.columns]
        df_disp = df_disp[prio + [c for c in df_disp.columns if c not in prio]]
        st.dataframe(df_disp.style.background_gradient(cmap="Blues", subset=["SalePrice"])
                     .format({"SalePrice":"{:,.0f}","GrLivArea":"{:,.0f}"}),
                     use_container_width=True, height=400)
        st.markdown(f"**{len(df_disp):,} lignes** affichées sur {len(df):,} filtrées.")
        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        buf = io.StringIO(); df.to_csv(buf, index=False)
        dl1.download_button("⬇️ Données filtrées (CSV)", buf.getvalue().encode(), f"filtered_{len(df)}.csv", "text/csv")
        buf2 = io.StringIO(); df.describe().to_csv(buf2)
        dl2.download_button("📈 Stats descriptives (CSV)", buf2.getvalue().encode(), "stats.csv", "text/csv")
        buf3 = io.StringIO()
        df.groupby("Neighborhood")["SalePrice"].agg(["count","mean","median"]).round(0).to_csv(buf3)
        dl3.download_button("📍 Résumé quartiers (CSV)", buf3.getvalue().encode(), "quartiers.csv", "text/csv")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — ENTRAÎNEMENT & PERFORMANCES                            ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 2":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 2 · Modélisation</div>
      <h1>Entraînement &amp;<br>Performances</h1>
      <p>Configurez, entraînez et comparez vos modèles de régression.</p>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.warning("⚠️ Aucun dataset chargé. Rendez-vous sur la **Page 1** pour charger vos données.", icon="⚠️")
        st.stop()

    df_raw = st.session_state.df_raw

    # ── Configuration ──────────────────────────────────────────────
    section("Configuration de l'entraînement")
    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        model_choice = st.selectbox(
            "🤖 Modèle",
            ["Régression Linéaire", "Random Forest", "Gradient Boosting", "Comparer les 3"],
            help="Choisissez le modèle à entraîner, ou comparez les 3 simultanément.",
        )
    with cfg2:
        test_size = st.slider("📊 Taille du jeu de test (%)", 10, 40, 20, step=5,
                              help="Pourcentage des données réservé à l'évaluation.")
    with cfg3:
        target_log = st.checkbox("🔁 Log-transformer la cible", value=True,
                                 help="Applique log1p(SalePrice) — recommandé pour normaliser la distribution.")
        remove_outliers = st.checkbox(
            "🧹 Supprimer les outliers (GrLivArea>4000 & Prix<300k)",
            value=True,
            help="Retire les ventes atypiques qui peuvent perturber l'entraînement.",
        )

    # Hyperparamètres avancés
    with st.expander("⚙️ Hyperparamètres avancés", expanded=False):
        ha1, ha2, ha3 = st.columns(3)
        with ha1:
            n_estimators = st.slider("n_estimators (RF / GB)", 50, 500, 300, step=50,
                                     help="Nombre d'arbres dans la forêt ou le boosting.")
        with ha2:
            max_depth = st.slider("max_depth (RF)", 2, 20, 0,
                                  help="Profondeur max des arbres RF (0 = illimité).")
        with ha3:
            lr_gb = st.slider("learning_rate (GB)", 0.01, 0.3, 0.05, step=0.01,
                              help="Taux d'apprentissage pour Gradient Boosting.")

    # ── Préparation des données ────────────────────────────────────
    @st.cache_data(show_spinner="Préparation des données…")
    def _prepare_cached(df_raw, test_sz, log_target, remove_outliers):
        return prepare_data(df_raw, test_sz, log_target, remove_outliers=remove_outliers)

    X_train, X_test, y_train, y_test, feat_cols, les, n_removed = _prepare_cached(
        df_raw, test_size, target_log, remove_outliers)

    if X_train is None:
        st.error("Colonne SalePrice introuvable.", icon="🚨"); st.stop()

    if remove_outliers and n_removed:
        st.info(f"🧹 {n_removed} lignes supprimées comme outliers (GrLivArea>4000 & Prix<300k).", icon="ℹ️")

    # ── Entraînement ───────────────────────────────────────────────
    section("Lancement de l'entraînement")

    if st.button("🚀 Entraîner le modèle", use_container_width=False):

        model_defs = get_model_defs(n_estimators, max_depth, lr_gb)
        to_train = list(model_defs.keys()) if model_choice == "Comparer les 3" else [model_choice]
        sub_defs = {k: model_defs[k] for k in to_train}

        st.info("Entraînement en cours…", icon="⏳")
        results = train_and_evaluate(X_train, y_train, X_test, y_test, sub_defs, target_log)

        # Sauvegarder le meilleur
        best_name = max(results, key=lambda k: results[k]["R²"])
        st.session_state.trained_model  = results[best_name]["model"]
        st.session_state.model_name     = best_name
        st.session_state.model_features = feat_cols
        st.session_state.label_encoders = les
        st.session_state.metrics        = results
        st.session_state.target_log     = target_log
        st.session_state.y_test         = y_test
        logger.info("Modèle entraîné : %s (R2=%s)", best_name, results[best_name]["R²"])
        st.success(f"✅ Entraînement terminé ! Meilleur modèle : **{best_name}** (R²={results[best_name]['R²']:.4f})")

    # ── Affichage résultats ────────────────────────────────────────
    if st.session_state.metrics:
        results   = st.session_state.metrics
        tgt_log   = st.session_state.get("target_log", True)
        y_te_orig = st.session_state.get("y_test", None)

        section("Tableau des performances")
        perf_df = pd.DataFrame({k: {kk: vv for kk,vv in v.items() if kk not in ["model","preds"]}
                                 for k,v in results.items()}).T
        st.dataframe(perf_df.style.background_gradient(cmap="Greens", subset=["R²"])
                                  .background_gradient(cmap="Reds_r", subset=["MAE"]),
                     use_container_width=True)

        # KPIs du meilleur
        best = max(results, key=lambda k: results[k]["R²"])
        b    = results[best]
        c1,c2,c3,c4 = st.columns(4)
        metric_card(c1, "Meilleur modèle", best.split()[0], "")
        metric_card(c2, "R²",   f"{b['R²']:.4f}", "coefficient déterm.")
        metric_card(c3, "MAE",  f"{b['MAE']:,.0f}$", "erreur absolue moy.")
        metric_card(c4, "RMSLE",f"{b['RMSLE']:.4f}", "erreur log")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Graphiques performances ────────────────────────────────
        section("Visualisation des performances")
        vt1, vt2, vt3 = st.tabs(["📊 Comparaison", "🎯 Réel vs Prédit", "📈 Feature Importance"])

        with vt1:
            fig, axes = plt.subplots(1,3,figsize=(15,5))
            for ax, met, color in zip(axes, ["R²","MAE","RMSLE"],
                                      ["#2ecc71","#e74c3c","#3498db"]):
                vals = [results[k][met] for k in results]
                bars = ax.bar(list(results.keys()), vals, color=color, edgecolor="white", width=0.5)
                ax.set_title(met, fontweight="600")
                ax.set_xticklabels(list(results.keys()), rotation=15, ha="right")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                            f"{v:,.4g}", ha="center", fontsize=9)
            plt.suptitle("Comparaison des modèles", fontsize=14, fontweight="600")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            # CV R²
            fig2, ax2 = plt.subplots(figsize=(8,4))
            means = [results[k]["CV_mean"] for k in results]
            stds  = [results[k]["CV_std"]  for k in results]
            ax2.bar(list(results.keys()), means, yerr=stds, capsize=6,
                    color=DARK_BLUE, edgecolor="white", alpha=0.85)
            ax2.set_title("Cross-Validation R² (5 folds)", fontweight="600")
            ax2.set_ylabel("R²"); ax2.set_xticklabels(list(results.keys()), rotation=15, ha="right")
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

        with vt2:
            sel_model = st.selectbox("Modèle à visualiser", list(results.keys()))
            m_preds = results[sel_model]["preds"]

            if y_te_orig is not None:
                y_real = np.expm1(y_te_orig) if tgt_log else y_te_orig
                y_pred = np.expm1(m_preds)   if tgt_log else m_preds

                fig, axes = plt.subplots(1,2,figsize=(14,5))
                axes[0].scatter(y_real, y_pred, alpha=0.4, s=18, color=DARK_BLUE)
                lims = [min(y_real.min(),y_pred.min()), max(y_real.max(),y_pred.max())]
                axes[0].plot(lims,lims,"r--",lw=1.5,label="Idéal")
                axes[0].set_xlabel("Prix réel ($)"); axes[0].set_ylabel("Prix prédit ($)")
                axes[0].set_title(f"{sel_model} — Réel vs Prédit", fontweight="600")
                axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x/1000:.0f}k"))
                axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x/1000:.0f}k"))
                axes[0].legend()

                residuals = y_real - y_pred
                sns.histplot(residuals, kde=True, ax=axes[1], color=CORAL)
                axes[1].axvline(0,color="black",ls="--"); axes[1].set_xlabel("Erreur ($)")
                axes[1].set_title("Distribution des résidus", fontweight="600")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with vt3:
            rf_models = {k:v for k,v in results.items()
                         if hasattr(v["model"],"feature_importances_")}
            if not rf_models:
                st.info("Feature importance disponible uniquement pour Random Forest et Gradient Boosting.", icon="ℹ️")
            else:
                sel_fi = st.selectbox("Modèle", list(rf_models.keys()), key="fi_sel")
                fi = pd.Series(rf_models[sel_fi]["model"].feature_importances_,
                               index=feat_cols).sort_values(ascending=False)[:20]
                fig, ax = plt.subplots(figsize=(10,6))
                fi.plot(kind="barh", color=DARK_BLUE, edgecolor="white", ax=ax)
                ax.invert_yaxis()
                ax.set_title(f"Top 20 Feature Importances — {sel_fi}", fontweight="600")
                ax.set_xlabel("Importance")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # ── Export modèle ──────────────────────────────────────────
        section("Export du modèle")
        buf_model = io.BytesIO()
        joblib.dump(st.session_state.trained_model, buf_model)
        st.download_button(
            "⬇️ Télécharger le modèle (.pkl)",
            data=buf_model.getvalue(),
            file_name=f"model_{st.session_state.model_name.replace(' ','_')}.pkl",
            mime="application/octet-stream",
            help="Sauvegarde le modèle entraîné au format joblib/pickle.",
        )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — INTERFACE DE PRÉDICTION                                ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "Page 3":

    st.markdown("""
    <div class="hero">
      <div class="badge">Page 3 · Prédiction</div>
      <h1>Interface de<br>Prédiction</h1>
      <p>Renseignez les caractéristiques d'un bien pour obtenir son prix estimé.</p>
    </div>""", unsafe_allow_html=True)

    # ── Vérifications prérequis ────────────────────────────────────
    has_model = st.session_state.trained_model is not None
    has_data  = st.session_state.df_raw is not None

    if not has_model:
        st.warning("⚠️ Aucun modèle entraîné. Rendez-vous sur la **Page 2** pour entraîner un modèle.", icon="⚠️")

    # Option charger modèle externe
    with st.expander("📦 Charger un modèle .pkl existant", expanded=not has_model):
        uploaded_model = st.file_uploader("Charger model.pkl", type=["pkl","joblib"],
                                          help="Chargez un modèle sauvegardé avec joblib.")
        if uploaded_model:
            try:
                st.session_state.trained_model  = joblib.load(uploaded_model)
                st.session_state.model_name     = "Modèle importé"
                has_model = True
                st.success("✅ Modèle chargé avec succès !")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

    if not has_model:
        st.stop()

    model = st.session_state.trained_model
    feat_cols = st.session_state.model_features
    les       = st.session_state.label_encoders
    tgt_log   = st.session_state.get("target_log", True)
    df_ref    = st.session_state.df_raw

    # ── Formulaire de saisie ───────────────────────────────────────
    section("Caractéristiques du bien")

    with st.form("prediction_form"):
        st.markdown("##### 🏗️ Informations principales")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        with r1c1:
            OverallQual = st.slider("⭐ Qualité générale", 1, 10, 7,
                                    help="Note globale de qualité des matériaux (1=très mauvais, 10=excellent).")
        with r1c2:
            GrLivArea = st.number_input("📐 Surface habitable (pi²)", 500, 6000, 1500, step=50,
                                        help="Surface totale habitable au-dessus du sol (GrLivArea).")
        with r1c3:
            TotalBsmtSF = st.number_input("🏚️ Surface sous-sol (pi²)", 0, 3000, 800, step=50,
                                          help="Surface totale du sous-sol (TotalBsmtSF).")
        with r1c4:
            GarageArea = st.number_input("🚗 Surface garage (pi²)", 0, 1500, 400, step=25,
                                         help="Surface du garage en pieds carrés.")

        st.markdown("##### 📅 Caractéristiques temporelles")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            YearBuilt = st.number_input("🏗️ Année de construction", 1850, 2024, 2000,
                                        help="Année originale de construction du bien.")
        with r2c2:
            YearRemodAdd = st.number_input("🔧 Année rénovation", 1850, 2024, 2005,
                                           help="Année de la dernière rénovation (= YearBuilt si jamais rénové).")
        with r2c3:
            OverallCond = st.slider("🔩 État général", 1, 10, 5,
                                    help="Note de l'état général du bien (1=très mauvais, 10=excellent).")

        st.markdown("##### 🛁 Pièces & équipements")
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        with r3c1:
            FullBath  = st.number_input("🚿 Salles de bain complètes", 0, 4, 2,
                                        help="Nombre de salles de bain complètes (baignoire + douche).")
        with r3c2:
            HalfBath  = st.number_input("🚽 Demi-salles de bain", 0, 3, 1,
                                        help="Nombre de demi-salles de bain (toilettes seules).")
        with r3c3:
            BedroomAbvGr = st.number_input("🛏️ Chambres", 0, 8, 3,
                                           help="Nombre de chambres au-dessus du sol.")
        with r3c4:
            TotRmsAbvGrd = st.number_input("🏠 Total pièces", 2, 15, 7,
                                           help="Nombre total de pièces au-dessus du sol (hors SDB).")

        st.markdown("##### 📍 Localisation & type")
        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1:
            neigh_opts = sorted(df_ref["Neighborhood"].unique()) if df_ref is not None else ["CollgCr"]
            Neighborhood = st.selectbox("📍 Quartier", neigh_opts,
                                        help="Quartier Ames où est situé le bien.")
        with r4c2:
            zone_opts = sorted(df_ref["MSZoning"].unique()) if df_ref is not None else ["RL"]
            MSZoning = st.selectbox("🗺️ Zone", zone_opts,
                                    help="Classification de zonage (RL=résidentiel faible densité).")
        with r4c3:
            LotArea = st.number_input("🌿 Surface terrain (pi²)", 1000, 100000, 8000, step=500,
                                      help="Surface totale du terrain en pieds carrés.")

        submitted = st.form_submit_button("🔮 Estimer le prix", use_container_width=True)

    # ── Prédiction ─────────────────────────────────────────────────
    if submitted:
        # Validation des entrées (évite les valeurs absurdes)
        if GrLivArea <= 0 or LotArea <= 0:
            st.error("La surface doit être un nombre positif.")
            st.stop()
        if TotalBsmtSF < 0 or GarageArea < 0:
            st.error("Les surfaces sous-sol / garage ne peuvent pas être négatives.")
            st.stop()
        if YearBuilt < 1800 or YearBuilt > 2025:
            st.error("L'année de construction doit être comprise entre 1800 et 2025.")
            st.stop()
        if Neighborhood not in df_ref["Neighborhood"].unique():
            st.error("Le quartier sélectionné n'est pas présent dans le dataset de référence.")
            st.stop()

        AgeLogement    = 2025 - YearBuilt
        SurfaceTotale  = GrLivArea + TotalBsmtSF
        NbSallesDeBain = FullBath + 0.5 * HalfBath

        # Construire le vecteur d'entrée
        input_dict = {
            "MSSubClass": 60, "MSZoning": MSZoning, "LotFrontage": 65,
            "LotArea": LotArea, "Street": "Pave", "Alley": "None",
            "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
            "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": Neighborhood,
            "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam",
            "HouseStyle": "2Story", "OverallQual": OverallQual, "OverallCond": OverallCond,
            "YearBuilt": YearBuilt, "YearRemodAdd": YearRemodAdd,
            "RoofStyle": "Gable", "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
            "MasVnrType": "None", "MasVnrArea": 0,
            "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc",
            "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No",
            "BsmtFinType1": "GLQ", "BsmtFinSF1": 500,
            "BsmtFinType2": "Unf", "BsmtFinSF2": 0,
            "BsmtUnfSF": 300, "TotalBsmtSF": TotalBsmtSF,
            "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y",
            "Electrical": "SBrkr", "1stFlrSF": GrLivArea//2, "2ndFlrSF": GrLivArea//2,
            "LowQualFinSF": 0, "GrLivArea": GrLivArea,
            "BsmtFullBath": 1, "BsmtHalfBath": 0,
            "FullBath": FullBath, "HalfBath": HalfBath,
            "BedroomAbvGr": BedroomAbvGr, "KitchenAbvGr": 1,
            "KitchenQual": "Gd", "TotRmsAbvGrd": TotRmsAbvGrd,
            "Functional": "Typ", "Fireplaces": 1, "FireplaceQu": "TA",
            "GarageType": "Attchd", "GarageYrBlt": YearBuilt,
            "GarageFinish": "RFn", "GarageCars": 2, "GarageArea": GarageArea,
            "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
            "WoodDeckSF": 0, "OpenPorchSF": 50, "EnclosedPorch": 0,
            "3SsnPorch": 0, "ScreenPorch": 0, "PoolArea": 0,
            "PoolQC": "None", "Fence": "None", "MiscFeature": "None",
            "MiscVal": 0, "MoSold": 6, "YrSold": 2024,
            "SaleType": "WD", "SaleCondition": "Normal",
            "AgeLogement": AgeLogement, "SurfaceTotale": SurfaceTotale,
            "NbSallesDeBain": NbSallesDeBain,
        }

        row = pd.DataFrame([input_dict])
        predicted_price = predict_from_input(model, row, feat_cols, les, tgt_log)
        logger.info(
            "Prediction demandée: %s → %s",
            {k: input_dict[k] for k in ['Neighborhood','OverallQual','GrLivArea','LotArea'] if k in input_dict},
            predicted_price,
        )

        # ── Résultat ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
          <div class="label">Prix estimé par {st.session_state.model_name}</div>
          <div class="price">${predicted_price:,.0f}</div>
          <div class="label" style="margin-top:0.6rem">
            Fourchette indicative :
            ${predicted_price*0.90:,.0f} — ${predicted_price*1.10:,.0f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Métriques contextuelles
        if df_ref is not None:
            med_global = df_ref["SalePrice"].median()
            delta_pct  = (predicted_price - med_global) / med_global * 100
            direction  = "au-dessus" if delta_pct >= 0 else "en-dessous"

            c1, c2, c3, c4 = st.columns(4)
            metric_card(c1, "Prix estimé",      f"${predicted_price:,.0f}", "")
            metric_card(c2, "Médiane dataset",   f"${med_global:,.0f}", "référence")
            metric_card(c3, "Écart à la méd.",   f"{abs(delta_pct):.1f}%", direction)
            metric_card(c4, "Age du logement",   f"{AgeLogement} ans", f"construit en {YearBuilt}")

            # Positionnement dans la distribution
            section("Positionnement dans le marché")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.hist(df_ref["SalePrice"], bins=50, color=DARK_BLUE, edgecolor="white", alpha=0.7, label="Distribution des prix")
            ax.axvline(predicted_price, color=GOLD, lw=2.5, ls="-",  label=f"Estimation : ${predicted_price:,.0f}")
            ax.axvline(med_global,      color=CORAL, lw=1.8, ls="--", label=f"Médiane : ${med_global:,.0f}")
            ax.set_xlabel("Prix de vente ($)", fontsize=12)
            ax.set_ylabel("Effectif", fontsize=12)
            ax.set_title("Position de votre bien dans le marché", fontweight="600", pad=14)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x/1000:.0f}k"))
            ax.legend(fontsize=10)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # Récapitulatif des inputs
        section("Récapitulatif des caractéristiques saisies")
        recap = {
            "Quartier": Neighborhood, "Zone": MSZoning,
            "Surface habitable": f"{GrLivArea} pi²", "Surface sous-sol": f"{TotalBsmtSF} pi²",
            "Surface terrain": f"{LotArea} pi²", "Garage": f"{GarageArea} pi²",
            "Qualité": f"{OverallQual}/10", "État": f"{OverallCond}/10",
            "Année construction": YearBuilt, "Âge": f"{AgeLogement} ans",
            "Chambres": BedroomAbvGr, "SDB": f"{NbSallesDeBain:.1f}",
            "Pièces totales": TotRmsAbvGrd,
        }
        rc1, rc2 = st.columns(2)
        items = list(recap.items())
        for i, (k,v) in enumerate(items):
            (rc1 if i < len(items)//2 else rc2).markdown(f"**{k}** : {v}")
