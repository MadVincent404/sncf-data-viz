import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SNCF Data Explorer",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS custom — couleurs SNCF
st.markdown("""
<style>
  :root { --sncf-red: #C0272D; --sncf-gray: #4A4A4A; }
  .stMetric { background: #fff; border-radius: 8px;
              border-left: 4px solid #C0272D; padding: 0.5rem 1rem; }
  h1, h2 { color: #C0272D !important; }
  .sidebar .sidebar-content { background: #F8F8F5; }
  div[data-testid="stSidebarNav"] a { color: #4A4A4A !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CACHE DONNÉES
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_frequentation():
    df = pd.read_csv("data/frequentation-gares.csv", sep=";")
    year_cols = [c for c in df.columns if c.startswith("Total Voyageurs 20") and "Non" not in c]
    id_cols = ["Nom de la gare", "Code UIC", "Direction Régionale Gares",
               "Segmentation DRG", "Segmentation Marketing"]
    df_long = df[id_cols + year_cols].melt(id_vars=id_cols, var_name="annee_str", value_name="voyageurs")
    df_long["annee"] = df_long["annee_str"].str.extract(r"(\d{4})").astype(int)
    df_long.rename(columns={
        "Nom de la gare": "gare", "Code UIC": "uic",
        "Direction Régionale Gares": "drg",
        "Segmentation DRG": "seg_drg",
        "Segmentation Marketing": "seg_mkt",
    }, inplace=True)
    return df_long, df.rename(columns={"Nom de la gare": "gare", "Code UIC": "uic"})

@st.cache_data(ttl=3600)
def load_tgv():
    df = pd.read_csv("data/regularite-mensuelle-tgv-aqst.csv", sep=";")
    df["date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")
    df["annee"] = df["date"].dt.year
    df["mois"]  = df["date"].dt.month
    df["ponctualite"] = (1 - df["Nombre de trains en retard à l'arrivée"]
                         / df["Nombre de circulations prévues"].replace(0, np.nan)) * 100
    return df

@st.cache_data(ttl=3600)
def load_transilien():
    df = pd.read_csv("data/comptage-voyageurs-trains-transilien.csv", sep=";")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df.rename(columns={"Nom Gare": "gare", "Somme de Montants": "montants",
                        "Type jour": "type_jour", "Tranche horaire": "tranche",
                        "Ligne": "ligne", "Annee": "annee"}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def load_gares():
    df = pd.read_csv("data/gares-de-voyageurs.csv", sep=";")
    df[["lat", "lon"]] = df["Position géographique"].str.split(",", expand=True).astype(float)
    df.rename(columns={"Nom": "gare", "Trigramme": "trigramme"}, inplace=True)
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("assets/téléchargement.jpg",
             width=120)
    st.markdown("##SNCF Data Explorer")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Accueil & KPIs", "EDA Fréquentation", "Transilien",
         "TGV Régularité", "Carte Interactive", ], # "Données Live"
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filtres globaux**")
    annee_range = st.slider("Période", 2015, 2024, (2019, 2024))
    st.markdown("---")
    st.caption("Données : SNCF Open Data · 2015–2024")


# ─────────────────────────────────────────────
# PAGE : ACCUEIL & KPIs
# ─────────────────────────────────────────────

if page == "Accueil & KPIs":
    st.title("Tableau de bord SNCF")
    st.markdown("*Analyse exploratoire complète du réseau ferroviaire français · 2015–2024*")

    freq_long, freq_wide = load_frequentation()
    tgv = load_tgv()

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    total_2024 = freq_long[freq_long.annee == 2024]["voyageurs"].sum()
    total_2019 = freq_long[freq_long.annee == 2019]["voyageurs"].sum()
    delta = (total_2024 - total_2019) / total_2019 * 100
    ponct = (1 - tgv["Nombre de trains en retard à l'arrivée"].sum()
             / tgv["Nombre de circulations prévues"].sum()) * 100

    col1.metric("Voyageurs 2024", f"{total_2024/1e6:.0f}M", f"{delta:+.1f}% vs 2019")
    col2.metric("Gares analysées",  f"{freq_wide['gare'].nunique():,}")
    col3.metric("Ponctualité TGV",  f"{ponct:.1f}%", "Moy. 2018–2024")
    col4.metric("Années de données", "10", "2015 → 2024")
    col5.metric("Datasets", "4", "Transilien · TGV · Gares")

    st.markdown("---")

    # Évolution nationale (Plotly)
    national = freq_long.groupby("annee")["voyageurs"].sum().reset_index()
    fig = px.area(national, x="annee", y="voyageurs",
                  title="Fréquentation nationale — impact COVID visible",
                  color_discrete_sequence=["#C0272D"])
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red", opacity=0.05,
                  annotation_text="COVID-19", annotation_position="top left")
    fig.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0",
                      yaxis_title="Voyageurs", xaxis_title="")
    fig.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig, use_container_width=True)

    # Top gares & segment
    col_a, col_b = st.columns(2)
    with col_a:
        if "Total Voyageurs 2024" in freq_wide.columns:
            top10 = freq_wide.nlargest(10, "Total Voyageurs 2024")[["gare", "Total Voyageurs 2024"]]
            fig2 = px.bar(top10.sort_values("Total Voyageurs 2024"),
                          x="Total Voyageurs 2024", y="gare",
                          orientation="h", title="Top 10 gares 2024",
                          color_discrete_sequence=["#C0272D"])
            fig2.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0")
            fig2.update_xaxes(tickformat=".2s")
            st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        seg = freq_long[freq_long.annee == 2024].groupby("seg_drg")["voyageurs"].sum().reset_index()
        fig3 = px.pie(seg, names="seg_drg", values="voyageurs",
                      title="Répartition par segment DRG — 2024",
                      color_discrete_sequence=["#C0272D", "#E87722", "#1A5276", "#0E7E6A", "#9B59B6"])
        fig3.update_layout(paper_bgcolor="#F5F5F0")
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE : EDA FRÉQUENTATION
# ─────────────────────────────────────────────

elif page == "EDA Fréquentation":
    st.title("Analyse de fréquentation")
    freq_long, freq_wide = load_frequentation()

    # Filtre période
    df = freq_long[freq_long.annee.between(*annee_range)]

    # Top gares sélectionnables
    top_gares = freq_wide.nlargest(50, "Total Voyageurs 2024")["gare"].tolist() \
        if "Total Voyageurs 2024" in freq_wide.columns else freq_wide["gare"].head(50).tolist()
    selected_gares = st.multiselect("Sélectionne des gares à comparer",
                                     top_gares, default=top_gares[:5])

    if selected_gares:
        df_sel = freq_long[freq_long.gare.isin(selected_gares)]
        fig = px.line(df_sel, x="annee", y="voyageurs", color="gare",
                      title="Évolution comparée — gares sélectionnées",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0")
        fig.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig, use_container_width=True)

    # Carte choroplèthe par DRG
    st.subheader("Fréquentation par direction régionale")
    drg_data = df.groupby(["drg", "annee"])["voyageurs"].sum().reset_index()
    drg_latest = drg_data[drg_data.annee == annee_range[1]]
    fig_drg = px.bar(drg_latest.sort_values("voyageurs", ascending=False),
                     x="drg", y="voyageurs",
                     color="voyageurs", color_continuous_scale="Reds",
                     title=f"Voyageurs par DRG — {annee_range[1]}")
    fig_drg.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0",
                           xaxis_tickangle=-35)
    fig_drg.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig_drg, use_container_width=True)

    # Tableau de données
    with st.expander("Voir les données brutes"):
        st.dataframe(df.sort_values(["annee", "voyageurs"], ascending=[False, False]).head(100),
                     use_container_width=True)


# ─────────────────────────────────────────────
# PAGE : TRANSILIEN
# ─────────────────────────────────────────────

elif page == "Transilien":
    st.title("Réseau Transilien")
    trans = load_transilien()

    col1, col2 = st.columns([1, 2])
    with col1:
        lignes_dispo = sorted(trans["ligne"].unique())
        lignes_sel = st.multiselect("Lignes", lignes_dispo, default=lignes_dispo[:5])

    df_t = trans[trans.ligne.isin(lignes_sel)] if lignes_sel else trans

    # Trafic par ligne
    by_line = df_t.groupby("ligne")["montants"].sum().sort_values(ascending=False).reset_index()
    fig_l = px.bar(by_line, x="ligne", y="montants",
                   title="Trafic total par ligne",
                   color="montants", color_continuous_scale="Reds")
    fig_l.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0")
    fig_l.update_yaxes(tickformat=".2s")
    st.plotly_chart(fig_l, use_container_width=True)

    # Heatmap heures × lignes
    order_tranches = ["Avant 6h", "De 6h à 9h", "De 9h à 11h", "De 11h à 14h",
                      "De 14h à 17h", "De 17h à 19h", "De 19h à 21h", "Après 21h"]
    pivot = df_t.groupby(["tranche", "ligne"])["montants"].sum().unstack(fill_value=0)
    existing = [t for t in order_tranches if t in pivot.index]
    pivot = pivot.reindex(existing)

    fig_h = px.imshow(pivot / 1e3, aspect="auto",
                      color_continuous_scale="YlOrRd",
                      title="Profil horaire par ligne (milliers de montages)",
                      labels={"x": "Ligne", "y": "Tranche horaire", "color": "k voy."})
    fig_h.update_layout(paper_bgcolor="#F5F5F0")
    st.plotly_chart(fig_h, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE : TGV RÉGULARITÉ
# ─────────────────────────────────────────────

elif page == "TGV Régularité":
    st.title("TGV — Ponctualité & Retards")
    tgv = load_tgv()
    df_tgv = tgv.dropna(subset=["date"])
    df_tgv = df_tgv[df_tgv.annee.between(*annee_range)]

    # Ponctualité mensuelle
    monthly = df_tgv.groupby("date").agg(
        circulations=("Nombre de circulations prévues", "sum"),
        retards=("Nombre de trains en retard à l'arrivée", "sum"),
    ).reset_index()
    monthly["ponctualite"] = (1 - monthly["retards"] / monthly["circulations"]) * 100

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["ponctualite"],
        fill="tonexty", fillcolor="rgba(192,39,45,0.1)",
        line=dict(color="#C0272D", width=2),
        name="Ponctualité"
    ))
    fig_p.add_hline(y=80, line_dash="dash", line_color="gray",
                    annotation_text="Seuil 80%")
    fig_p.update_layout(
        title="Taux de ponctualité mensuel TGV",
        paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0",
        yaxis_title="%", yaxis_range=[60, 100]
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # Causes de retard
    col1, col2 = st.columns(2)
    cause_cols = {
        "Externes":        "Prct retard pour causes externes",
        "Infrastructure":  "Prct retard pour cause infrastructure",
        "Gestion trafic":  "Prct retard pour cause gestion trafic",
        "Matériel":        "Prct retard pour cause matériel roulant",
        "Gestion gare":    "Prct retard pour cause gestion en gare et réutilisation de matériel",
        "Voyageurs":       "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
    }
    with col1:
        means = {k: df_tgv[v].mean() for k, v in cause_cols.items() if v in df_tgv.columns}
        fig_pie = px.pie(names=list(means.keys()), values=list(means.values()),
                         title="Causes de retard",
                         color_discrete_sequence=["#C0272D","#E87722","#1A5276","#0E7E6A","#9B59B6","#27AE60"])
        fig_pie.update_layout(paper_bgcolor="#F5F5F0")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        top_ret = (df_tgv.groupby(["Gare de départ", "Gare d'arrivée"])
                   .agg(retard=("Retard moyen de tous les trains à l'arrivée", "mean"))
                   .reset_index().nlargest(12, "retard"))
        top_ret["liaison"] = top_ret["Gare de départ"].str[:10] + "→" + top_ret["Gare d'arrivée"].str[:10]
        fig_bar = px.bar(top_ret.sort_values("retard"),
                         x="retard", y="liaison", orientation="h",
                         title="Top 12 liaisons — retard moyen",
                         color="retard", color_continuous_scale="Reds")
        fig_bar.update_layout(paper_bgcolor="#F5F5F0", plot_bgcolor="#F5F5F0")
        st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE : CARTE INTERACTIVE
# ─────────────────────────────────────────────

elif page == "Carte Interactive":
    st.title("Carte des gares SNCF")

    map_file = Path("outputs/map_sncf_interactive.html")
    if map_file.exists():
        with open(map_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
        st.caption("Utilise les contrôles en haut à droite pour changer de couche.")
    else:
        st.warning("Carte non générée. Lance d'abord : `python map_sncf.py`")
        st.code("python map_sncf.py", language="bash")

        # Carte de secours avec Plotly
        gares = load_gares()
        freq_long, freq_wide = load_frequentation()
        if "Total Voyageurs 2024" in freq_wide.columns:
            merged = gares.merge(freq_wide[["gare", "Total Voyageurs 2024"]], on="gare", how="left")
            merged["Total Voyageurs 2024"] = merged["Total Voyageurs 2024"].fillna(0)
            fig_map = px.scatter_mapbox(
                merged.dropna(subset=["lat", "lon"]),
                lat="lat", lon="lon", hover_name="gare",
                size="Total Voyageurs 2024", size_max=25,
                color="Total Voyageurs 2024", color_continuous_scale="Reds",
                zoom=5, center={"lat": 46.8, "lon": 2.3},
                mapbox_style="carto-positron",
                title="Gares SNCF — fréquentation 2024 (carte de secours Plotly)"
            )
            fig_map.update_layout(paper_bgcolor="#F5F5F0", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_map, use_container_width=True)

