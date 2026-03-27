import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG GLOBALE — palette SNCF + style épuré
# ─────────────────────────────────────────────

SNCF_RED   = "#C0272D"
SNCF_GRAY  = "#4A4A4A"
SNCF_LIGHT = "#F5F5F0"
ACCENT     = "#E87722"   # orange chaud pour les highlights
BLUE       = "#1A5276"
TEAL       = "#0E7E6A"

PALETTE_LINES = {
    "A": "#E30613", "B": "#007DC5", "C": "#F1A020",
    "D": "#007A53", "E": "#9F4098", "H": "#6E2585",
    "J": "#CDAE00", "K": "#7B4397", "L": "#9C1A17",
    "N": "#005E99", "P": "#E05206", "R": "#E6A609",
    "U": "#C4003A",
}

plt.rcParams.update({
    "figure.facecolor":  SNCF_LIGHT,
    "axes.facecolor":    SNCF_LIGHT,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         True,
    "grid.color":        "#DDDDDD",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   SNCF_GRAY,
    "axes.labelcolor":   SNCF_GRAY,
    "xtick.color":       SNCF_GRAY,
    "ytick.color":       SNCF_GRAY,
})

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────

def load_data() -> dict[str, pd.DataFrame]:
    """Charge et nettoie les 4 datasets SNCF."""
    dfs = {}

    # --- Transilien ---
    df = pd.read_csv("data/comptage-voyageurs-trains-transilien.csv", sep=";")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    df.rename(columns={"Nom Gare": "gare", "Code Gare": "code_gare",
                        "Type jour": "type_jour", "Annee": "annee",
                        "Ligne": "ligne", "Axe": "axe",
                        "Tranche horaire": "tranche", "Somme de Montants": "montants"}, inplace=True)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["type_jour"] = df["type_jour"].str.strip()
    dfs["transilien"] = df

    # --- Fréquentation gares ---
    df = pd.read_csv("data/frequentation-gares.csv", sep=";")
    # Colonnes voyageurs 2015→2024 au format long
    year_cols = [c for c in df.columns if c.startswith("Total Voyageurs 20") and "Non" not in c]
    id_cols = ["Nom de la gare", "Code UIC", "Direction Régionale Gares",
               "Segmentation DRG", "Segmentation Marketing"]
    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols, var_name="annee_str", value_name="voyageurs"
    )
    df_long["annee"] = df_long["annee_str"].str.extract(r"(\d{4})").astype(int)
    df_long.rename(columns={"Nom de la gare": "gare", "Code UIC": "uic",
                             "Direction Régionale Gares": "drg",
                             "Segmentation DRG": "seg_drg",
                             "Segmentation Marketing": "seg_mkt"}, inplace=True)
    dfs["freq"] = df_long
    dfs["freq_wide"] = df.rename(columns={"Nom de la gare": "gare", "Code UIC": "uic"})

    # --- Gares de voyageurs ---
    df = pd.read_csv("data/gares-de-voyageurs.csv", sep=";")
    df[["lat", "lon"]] = df["Position géographique"].str.split(",", expand=True).astype(float)
    df.rename(columns={"Nom": "gare", "Trigramme": "trigramme",
                        "Segment(s) DRG": "segment", "Code(s) UIC": "uic"}, inplace=True)
    dfs["gares"] = df

    # --- TGV régularité ---
    df = pd.read_csv("data/regularite-mensuelle-tgv-aqst.csv", sep=";")
    df["date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")
    df["annee"] = df["date"].dt.year
    df["mois"]  = df["date"].dt.month
    df["taux_ponctualite"] = 100 - (
        df["Nombre de trains en retard à l'arrivée"] /
        df["Nombre de circulations prévues"].replace(0, np.nan) * 100
    )
    df["taux_annulation"] = (
        df["Nombre de trains annulés"] /
        df["Nombre de circulations prévues"].replace(0, np.nan) * 100
    )
    dfs["tgv"] = df

    print(" Données chargées :")
    for k, v in dfs.items():
        print(f"   {k:12s} → {v.shape[0]:>6,} lignes × {v.shape[1]} colonnes")
    return dfs


# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SNCF_LIGHT)
    plt.close(fig)
    print(f"  {path}")


def _title_annotation(ax, text: str):
    """Petit sous-titre grisé sous le titre principal."""
    ax.annotate(text, xy=(0, 1.02), xycoords="axes fraction",
                fontsize=9, color="#888888", ha="left")


def _add_value_labels(ax, fmt="{:.0f}", fontsize=9, color=SNCF_GRAY, offset=3):
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0:
            ax.text(patch.get_x() + patch.get_width() / 2,
                    h + offset, fmt.format(h),
                    ha="center", va="bottom", fontsize=fontsize, color=color)


# ─────────────────────────────────────────────
# 3. VUE D'ENSEMBLE — rapport qualité des données
# ─────────────────────────────────────────────

def plot_data_quality(dfs: dict) -> None:
    """Heatmap des valeurs manquantes + statistiques descriptives."""
    print("\n[1/7] Qualité des données")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Qualité des données — taux de complétude par dataset",
                 fontsize=15, fontweight="bold", color=SNCF_GRAY, y=1.02)

    datasets = {
        "Transilien": dfs["transilien"],
        "Fréquentation": dfs["freq_wide"],
        "Gares": dfs["gares"],
        "TGV Régularité": dfs["tgv"],
    }

    for ax, (name, df) in zip(axes, datasets.items()):
        completeness = (1 - df.isnull().mean()) * 100
        colors = [TEAL if v == 100 else ACCENT if v >= 90 else SNCF_RED
                  for v in completeness.values]
        bars = ax.barh(range(len(completeness)), completeness.values,
                       color=colors, height=0.7)
        ax.set_yticks(range(len(completeness)))
        ax.set_yticklabels([c[:22] for c in completeness.index], fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axvline(100, color="#CCCCCC", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, completeness.values):
            ax.text(min(val + 1.5, 105), bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=7.5, color=SNCF_GRAY)

    legend_patches = [
        mpatches.Patch(color=TEAL,     label="100% complet"),
        mpatches.Patch(color=ACCENT,   label="≥ 90%"),
        mpatches.Patch(color=SNCF_RED, label="< 90%"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "01_data_quality")


# ─────────────────────────────────────────────
# 4. FRÉQUENTATION — évolution 2015-2024
# ─────────────────────────────────────────────

def plot_frequentation_evolution(dfs: dict) -> None:
    """Évolution nationale + impact COVID clairement mis en évidence."""
    print("[2/7] Évolution fréquentation nationale")
    df = dfs["freq"]
    national = df.groupby("annee")["voyageurs"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Zone COVID
    ax.axvspan(2019.5, 2021.5, alpha=0.08, color=SNCF_RED, label="Période COVID")
    ax.annotate(" COVID-19\nChute -72%", xy=(2020, national.loc[national.annee==2020, "voyageurs"].values[0]),
                xytext=(2020.4, national["voyageurs"].max() * 0.7),
                arrowprops=dict(arrowstyle="->", color=SNCF_RED, lw=1.5),
                fontsize=10, color=SNCF_RED, fontweight="bold")

    # Ligne principale
    ax.plot(national["annee"], national["voyageurs"] / 1e6,
            color=SNCF_RED, linewidth=2.5, zorder=5)
    ax.fill_between(national["annee"], national["voyageurs"] / 1e6,
                    alpha=0.12, color=SNCF_RED)
    ax.scatter(national["annee"], national["voyageurs"] / 1e6,
               color=SNCF_RED, zorder=6, s=50)

    # Annotations min/max
    idx_max = national["voyageurs"].idxmax()
    idx_min = national["voyageurs"].idxmin()
    for idx, label, offset in [(idx_max, "Record", 20), (idx_min, "Creux COVID", -40)]:
        row = national.iloc[idx]
        ax.annotate(f"{label}\n{row.voyageurs/1e6:.0f}M",
                    xy=(row.annee, row.voyageurs / 1e6),
                    xytext=(row.annee + 0.3, row.voyageurs / 1e6 + offset),
                    fontsize=9, color=SNCF_GRAY,
                    arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.8))

    ax.set_title("Fréquentation totale des gares SNCF — 2015 à 2024")
    _title_annotation(ax, "Voyageurs annuels (millions) · Toutes gares confondues")
    ax.set_xlabel("")
    ax.set_ylabel("Voyageurs (millions)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xticks(national["annee"])
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, "02_frequentation_nationale")


def plot_frequentation_top_gares(dfs: dict) -> None:
    """Top 20 gares + évolution comparative 2019 vs 2024."""
    print("[3/7] Top gares & comparaison 2019→2024")
    wide = dfs["freq_wide"]
    year_cols_v = [c for c in wide.columns if c.startswith("Total Voyageurs 20") and "Non" not in c]
    wide["total_all"] = wide[year_cols_v].sum(axis=1)
    top20 = wide.nlargest(20, "total_all")[["gare", "Total Voyageurs 2019", "Total Voyageurs 2024"]].copy()
    top20 = top20.sort_values("Total Voyageurs 2024")
    top20["delta_pct"] = (top20["Total Voyageurs 2024"] - top20["Total Voyageurs 2019"]) \
                          / top20["Total Voyageurs 2019"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Graphe 1 : barres horizontales 2024
    bars = ax1.barh(top20["gare"], top20["Total Voyageurs 2024"] / 1e6,
                    color=SNCF_RED, alpha=0.85, height=0.7)
    ax1.set_title("Top 20 gares — fréquentation 2024")
    _title_annotation(ax1, "Millions de voyageurs")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    for bar, val in zip(bars, top20["Total Voyageurs 2024"] / 1e6):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}M", va="center", fontsize=8.5, color=SNCF_GRAY)

    # Graphe 2 : évolution 2019 → 2024 (delta %)
    colors = [TEAL if d >= 0 else SNCF_RED for d in top20["delta_pct"]]
    ax2.barh(top20["gare"], top20["delta_pct"], color=colors, alpha=0.85, height=0.7)
    ax2.axvline(0, color=SNCF_GRAY, linewidth=1)
    ax2.set_title("Variation 2019 → 2024 (%)")
    _title_annotation(ax2, "Impact COVID + rebond · positif = hausse vs pré-COVID")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%"))

    patches = [mpatches.Patch(color=TEAL, label="Rebond ≥ 2019"),
               mpatches.Patch(color=SNCF_RED, label="Encore sous 2019")]
    ax2.legend(handles=patches, loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    _save(fig, "03_top_gares")


# ─────────────────────────────────────────────
# 5. TRANSILIEN — analyse par ligne & tranche horaire
# ─────────────────────────────────────────────

def plot_transilien_lines(dfs: dict) -> None:
    """Trafic par ligne + profil horaire (heatmap)."""
    print("[4/7] Transilien — lignes & heures de pointe")
    df = dfs["transilien"]

    # --- Trafic total par ligne ---
    by_line = df.groupby("ligne")["montants"].sum().sort_values(ascending=False)

    # --- Heatmap tranche × ligne ---
    order_tranches = ["Avant 6h", "De 6h à 9h", "De 9h à 11h", "De 11h à 14h",
                      "De 14h à 17h", "De 17h à 19h", "De 19h à 21h", "Après 21h"]
    pivot = df.groupby(["tranche", "ligne"])["montants"].sum().unstack(fill_value=0)
    existing = [t for t in order_tranches if t in pivot.index]
    pivot = pivot.reindex(existing)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1, 1.6]})

    # Barres par ligne
    colors_l = [PALETTE_LINES.get(l, SNCF_GRAY) for l in by_line.index]
    ax1.barh(by_line.index, by_line.values / 1e6, color=colors_l, height=0.7)
    ax1.set_title("Trafic total par ligne Transilien")
    _title_annotation(ax1, "Somme des montages (millions)")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Heatmap
    sns.heatmap(pivot / 1e3, ax=ax2, cmap="YlOrRd", linewidths=0.4,
                linecolor="#EEEEEE", annot=True, fmt=".0f",
                annot_kws={"size": 8}, cbar_kws={"label": "Milliers de voyageurs"})
    ax2.set_title("Profil horaire par ligne")
    _title_annotation(ax2, "Milliers de montages · toutes dates confondues")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    _save(fig, "04_transilien_lignes")


def plot_transilien_jour(dfs: dict) -> None:
    """Comparaison semaine / samedi / dimanche par tranche horaire."""
    print("[5/7] Transilien — profil LUN-VEN vs week-end")
    df = dfs["transilien"]

    order_tranches = ["Avant 6h", "De 6h à 9h", "De 9h à 11h", "De 11h à 14h",
                      "De 14h à 17h", "De 17h à 19h", "De 19h à 21h", "Après 21h"]

    # Regrouper LUNDI→VENDREDI si présents, sinon garder les libellés bruts
    mapping = {"LUN": "Semaine", "MAR": "Semaine", "MER": "Semaine",
               "JEU": "Semaine", "VEN": "Semaine", "SAM": "Samedi", "DIM": "Dimanche"}
    df["jour_cat"] = df["type_jour"].map(mapping).fillna(df["type_jour"])

    pivot = (df.groupby(["tranche", "jour_cat"])["montants"]
               .sum().unstack(fill_value=0))
    existing = [t for t in order_tranches if t in pivot.index]
    pivot = pivot.reindex(existing)

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(pivot))
    width = 0.26
    colors_map = {"Semaine": SNCF_RED, "Samedi": ACCENT, "Dimanche": BLUE}

    for i, col in enumerate(pivot.columns):
        if col in colors_map:
            offset = (i - len(pivot.columns) / 2 + 0.5) * width
            ax.bar(x + offset, pivot[col] / 1e3, width,
                   label=col, color=colors_map.get(col, SNCF_GRAY), alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title("Profil horaire Transilien — Semaine vs Week-end")
    _title_annotation(ax, "Milliers de montages par tranche horaire")
    ax.set_ylabel("Milliers de voyageurs")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.legend(frameon=False)

    fig.tight_layout()
    _save(fig, "05_transilien_jours")


# ─────────────────────────────────────────────
# 6. TGV — régularité & causes de retard
# ─────────────────────────────────────────────

def plot_tgv_regularite(dfs: dict) -> None:
    """Taux de ponctualité mensuel + évolution par axe."""
    print("[6/7] TGV — ponctualité & causes de retard")
    df = dfs["tgv"].dropna(subset=["date"])

    # Ponctualité mensuelle nationale
    monthly = df.groupby("date").agg(
        circulations=("Nombre de circulations prévues", "sum"),
        retards=("Nombre de trains en retard à l'arrivée", "sum"),
        annulations=("Nombre de trains annulés", "sum")
    ).reset_index()
    monthly["ponctualite"] = (1 - monthly["retards"] / monthly["circulations"]) * 100

    cause_cols = {
        "Externes":          "Prct retard pour causes externes",
        "Infrastructure":    "Prct retard pour cause infrastructure",
        "Gestion trafic":    "Prct retard pour cause gestion trafic",
        "Matériel roulant":  "Prct retard pour cause matériel roulant",
        "Gestion gare":      "Prct retard pour cause gestion en gare et réutilisation de matériel",
        "Voyageurs":         "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
    }
    causes_mean = {k: df[v].mean() for k, v in cause_cols.items()}

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # --- Courbe ponctualité ---
    ax1.plot(monthly["date"], monthly["ponctualite"],
             color=SNCF_RED, linewidth=1.8, alpha=0.9)
    ax1.fill_between(monthly["date"], monthly["ponctualite"], 80,
                     where=monthly["ponctualite"] < 80,
                     color=SNCF_RED, alpha=0.18, label="< 80% ponctualité")
    ax1.axhline(80, color=SNCF_RED, linewidth=1, linestyle="--", alpha=0.5)
    ax1.axhspan(2020, 2021, alpha=0.0)  # placeholder
    ax1.set_ylim(60, 100)
    ax1.set_title("Taux de ponctualité mensuel TGV (toutes liaisons)")
    _title_annotation(ax1, "% trains arrivés à l'heure · 2018–2024")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax1.legend(frameon=False)

    # Grève 2019 annotation
    greve_date = pd.Timestamp("2019-12-01")
    if monthly["date"].min() <= greve_date <= monthly["date"].max():
        ponct_greve = monthly.loc[monthly["date"].dt.month == 12, "ponctualite"]
        if not ponct_greve.empty:
            ax1.annotate("Grèves déc. 2019", xy=(greve_date, ponct_greve.min()),
                        xytext=(pd.Timestamp("2020-06-01"), 68),
                        arrowprops=dict(arrowstyle="->", color=SNCF_GRAY, lw=1),
                        fontsize=9, color=SNCF_GRAY)

    # --- Causes de retard (pie) ---
    colors_causes = [SNCF_RED, ACCENT, BLUE, TEAL, "#9B59B6", "#27AE60"]
    wedges, texts, autotexts = ax2.pie(
        list(causes_mean.values()),
        labels=list(causes_mean.keys()),
        colors=colors_causes,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.78, labeldistance=1.1,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for t in texts: t.set_fontsize(8.5)
    for at in autotexts: at.set_fontsize(8); at.set_color("white")
    ax2.set_title("Répartition des causes de retard")

    # --- Top 15 liaisons avec le plus de retards ---
    top_retards = (df.groupby(["Gare de départ", "Gare d'arrivée"])
                     .agg(retard_moy=("Retard moyen de tous les trains à l'arrivée", "mean"))
                     .reset_index()
                     .nlargest(15, "retard_moy"))
    top_retards["liaison"] = (top_retards["Gare de départ"].str[:12]
                               + " → " + top_retards["Gare d'arrivée"].str[:12])
    top_retards = top_retards.sort_values("retard_moy")
    colors_bar = [SNCF_RED if v > top_retards["retard_moy"].median() else ACCENT
                  for v in top_retards["retard_moy"]]
    ax3.barh(top_retards["liaison"], top_retards["retard_moy"],
             color=colors_bar, height=0.7)
    ax3.set_title("Top 15 liaisons — retard moyen à l'arrivée")
    _title_annotation(ax3, "Minutes · moyenne sur toute la période")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f} min"))
    ax3.tick_params(axis="y", labelsize=8)

    fig.suptitle("Analyse TGV — Ponctualité & Retards", fontsize=16,
                 fontweight="bold", color=SNCF_GRAY, y=1.01)
    _save(fig, "06_tgv_regularite")


# ─────────────────────────────────────────────
# 7. SYNTHÈSE — dashboard résumé 1 page
# ─────────────────────────────────────────────

def plot_dashboard_synthese(dfs: dict) -> None:
    """KPI globaux + mini-graphes — vue exécutive pour le portfolio."""
    print("[7/7] Dashboard synthèse")
    freq  = dfs["freq"]
    tgv   = dfs["tgv"].dropna(subset=["date"])
    trans = dfs["transilien"]

    # KPIs
    total_2024 = freq[freq.annee == 2024]["voyageurs"].sum()
    total_2023 = freq[freq.annee == 2023]["voyageurs"].sum()
    total_2019 = freq[freq.annee == 2019]["voyageurs"].sum()
    delta_covid = (total_2024 - total_2019) / total_2019 * 100

    ponct_mean = (1 - tgv["Nombre de trains en retard à l'arrivée"].sum()
                  / tgv["Nombre de circulations prévues"].sum()) * 100
    annul_mean = tgv["taux_annulation"].mean()
    nb_gares = dfs["gares"].shape[0]

    fig = plt.figure(figsize=(18, 11), facecolor=SNCF_LIGHT)
    fig.suptitle("Dashboard SNCF — Vue d'ensemble des données",
                 fontsize=18, fontweight="bold", color=SNCF_GRAY, y=0.98)

    # Ligne de KPIs (en haut)
    kpi_data = [
        ("Gares actives", f"{nb_gares:,}", "Réseau voyageurs"),
        ("Voyageurs 2024", f"{total_2024/1e6:.0f}M", "Total toutes gares"),
        ("Vs pré-COVID 2019", f"{delta_covid:+.1f}%",
         "Rebond" if delta_covid >= 0 else "Déficit persistant"),
        ("Ponctualité TGV", f"{ponct_mean:.1f}%", "Moyenne 2018–2024"),
        ("Taux annulation", f"{annul_mean:.1f}%", "Moyenne 2018–2024"),
    ]

    kpi_colors = [BLUE, TEAL, TEAL if delta_covid >= 0 else SNCF_RED,
                  TEAL if ponct_mean > 80 else ACCENT, ACCENT]

    for i, (label, value, sub) in enumerate(kpi_data):
        ax_kpi = fig.add_axes([0.04 + i * 0.19, 0.82, 0.17, 0.13])
        ax_kpi.set_facecolor("white")
        for spine in ax_kpi.spines.values():
            spine.set_visible(False)
        ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
        ax_kpi.text(0.5, 0.70, value, transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=22, fontweight="bold",
                    color=kpi_colors[i])
        ax_kpi.text(0.5, 0.35, label, transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=9.5, color=SNCF_GRAY, fontweight="bold")
        ax_kpi.text(0.5, 0.12, sub, transform=ax_kpi.transAxes,
                    ha="center", va="center", fontsize=8, color="#AAAAAA")
        ax_kpi.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02", transform=ax_kpi.transAxes,
            facecolor="white", edgecolor=kpi_colors[i], linewidth=1.5, zorder=0))

    # Mini graphe 1 : évolution nationale
    ax_nat = fig.add_subplot(3, 3, 4)
    nat = freq.groupby("annee")["voyageurs"].sum() / 1e6
    ax_nat.plot(nat.index, nat.values, color=SNCF_RED, linewidth=2)
    ax_nat.fill_between(nat.index, nat.values, alpha=0.1, color=SNCF_RED)
    ax_nat.set_title("Fréquentation nationale", fontsize=10, fontweight="bold")
    ax_nat.set_ylabel("Millions"); ax_nat.tick_params(labelsize=8)

    # Mini graphe 2 : top 10 gares 2024
    ax_top = fig.add_subplot(3, 3, 5)
    wide = dfs["freq_wide"]
    if "Total Voyageurs 2024" in wide.columns:
        top10 = wide.nlargest(10, "Total Voyageurs 2024")[["gare", "Total Voyageurs 2024"]]
        ax_top.barh(top10["gare"].str[:20], top10["Total Voyageurs 2024"] / 1e6,
                    color=SNCF_RED, alpha=0.8, height=0.7)
    ax_top.set_title("Top 10 gares 2024", fontsize=10, fontweight="bold")
    ax_top.tick_params(labelsize=7.5)
    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Mini graphe 3 : ponctualité TGV annuelle
    ax_ponct = fig.add_subplot(3, 3, 6)
    yearly_p = tgv.groupby("annee").apply(
        lambda x: (1 - x["Nombre de trains en retard à l'arrivée"].sum()
                   / x["Nombre de circulations prévues"].sum()) * 100
    ).reset_index(name="ponctualite")
    bar_colors = [TEAL if p >= 80 else SNCF_RED for p in yearly_p["ponctualite"]]
    ax_ponct.bar(yearly_p["annee"], yearly_p["ponctualite"],
                 color=bar_colors, alpha=0.85)
    ax_ponct.axhline(80, color=SNCF_RED, linewidth=1, linestyle="--", alpha=0.6)
    ax_ponct.set_ylim(70, 100)
    ax_ponct.set_title("Ponctualité TGV / an", fontsize=10, fontweight="bold")
    ax_ponct.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_ponct.tick_params(labelsize=8)

    # Mini graphe 4 : Transilien par tranche
    ax_tr = fig.add_subplot(3, 3, 7)
    order_tranches = ["Avant 6h", "De 6h à 9h", "De 9h à 11h", "De 11h à 14h",
                      "De 14h à 17h", "De 17h à 19h", "De 19h à 21h", "Après 21h"]
    tr_h = trans.groupby("tranche")["montants"].sum().reindex(
        [t for t in order_tranches if t in trans["tranche"].unique()])
    ax_tr.bar(range(len(tr_h)), tr_h.values / 1e3,
              color=[SNCF_RED if v == tr_h.max() else SNCF_GRAY for v in tr_h.values],
              alpha=0.85)
    ax_tr.set_xticks(range(len(tr_h)))
    ax_tr.set_xticklabels([t.replace("De ", "").replace("h", "h\n") for t in tr_h.index],
                           fontsize=6.5, rotation=0)
    ax_tr.set_title("Transilien — heures de pointe", fontsize=10, fontweight="bold")
    ax_tr.set_ylabel("Milliers"); ax_tr.tick_params(labelsize=8)

    # Mini graphe 5 : répartition segment DRG
    ax_seg = fig.add_subplot(3, 3, 8)
    seg = freq[freq.annee == 2024].groupby("seg_drg")["voyageurs"].sum().sort_values(ascending=False)
    ax_seg.bar(seg.index, seg.values / 1e6,
               color=[SNCF_RED, ACCENT, BLUE, TEAL, "#9B59B6"][:len(seg)], alpha=0.85)
    ax_seg.set_title("Voyageurs 2024 par segment", fontsize=10, fontweight="bold")
    ax_seg.set_ylabel("Millions"); ax_seg.tick_params(labelsize=9)
    ax_seg.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # Mini graphe 6 : causes retard TGV donut
    ax_cause = fig.add_subplot(3, 3, 9)
    cause_cols = {
        "Externes": "Prct retard pour causes externes",
        "Infra": "Prct retard pour cause infrastructure",
        "Trafic": "Prct retard pour cause gestion trafic",
        "Matériel": "Prct retard pour cause matériel roulant",
        "Gare/Réutilisation": "Prct retard pour cause gestion en gare et réutilisation de matériel",
        "Voyageurs": "Prct retard pour cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",
    }
    means = [tgv[v].mean() for v in cause_cols.values()]
    colors_c = [SNCF_RED, ACCENT, BLUE, TEAL, "#9B59B6", "#27AE60"]
    wedges, _ = ax_cause.pie(means, colors=colors_c, startangle=140,
                              wedgeprops={"edgecolor": "white", "linewidth": 1.2, "width": 0.55})
    ax_cause.legend(wedges, cause_cols.keys(), loc="lower center",
                    fontsize=6.5, ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.22))
    ax_cause.set_title("Causes de retard TGV", fontsize=10, fontweight="bold")

    plt.subplots_adjust(left=0.05, right=0.97, top=0.80, bottom=0.07,
                        hspace=0.55, wspace=0.35)
    _save(fig, "00_dashboard_synthese")


# ─────────────────────────────────────────────
# 8. BONUS — corrélation multi-datasets
# ─────────────────────────────────────────────

def plot_correlation_matrix(dfs: dict) -> None:
    """Matrice de corrélation TGV : retards, annulations, causes."""
    print("[BONUS] Matrice de corrélation TGV")
    df = dfs["tgv"].select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    # Sélection des colonnes pertinentes
    cols = [
        "Nombre de circulations prévues",
        "Nombre de trains annulés",
        "Nombre de trains en retard à l'arrivée",
        "Retard moyen de tous les trains à l'arrivée",
        "Nombre trains en retard > 15min",
        "Nombre trains en retard > 30min",
        "Nombre trains en retard > 60min",
        "Prct retard pour causes externes",
        "Prct retard pour cause infrastructure",
        "Prct retard pour cause gestion trafic",
        "Prct retard pour cause matériel roulant",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    short_names = {c: c.replace("Nombre de trains en retard", "Nb retard")
                      .replace("Prct retard pour cause", "Cause")
                      .replace("Prct retard pour causes", "Cause")
                      .replace("Nombre de circulations prévues", "Circulations")
                      .replace("Retard moyen de tous les trains à l'arrivée", "Retard moy.")
                      .replace("Nombre trains en retard", "Nb retard")
                      .replace(" à l'arrivée", "")
                      for c in cols}
    corr.rename(index=short_names, columns=short_names, inplace=True)

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5, linecolor="#EEEEEE",
                cbar_kws={"label": "Corrélation de Pearson"})
    ax.set_title("Corrélation entre indicateurs TGV")
    _title_annotation(ax, "Triangle inférieur · 1 = corrélation parfaite · -1 = anti-corrélation")
    ax.tick_params(axis="x", rotation=40, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    fig.tight_layout()
    _save(fig, "07_correlation_tgv")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_full_eda():
    print("=" * 60)
    print("  EDA SNCF — démarrage")
    print("=" * 60)

    dfs = load_data()

    plot_data_quality(dfs)
    plot_frequentation_evolution(dfs)
    plot_frequentation_top_gares(dfs)
    plot_transilien_lines(dfs)
    plot_transilien_jour(dfs)
    plot_tgv_regularite(dfs)
    plot_dashboard_synthese(dfs)
    plot_correlation_matrix(dfs)

    print(f"\nEDA terminé — {len(list(OUTPUT_DIR.glob('*.png')))} graphiques dans {OUTPUT_DIR}/")
    print("\n  Prochaines étapes suggérées :")
    print("  1. python map_sncf.py         → carte SIG interactive Folium")
    print("  2. python api_sncf.py          → intégration API temps réel")
    print("  3. streamlit run app.py        → dashboard Streamlit")


if __name__ == "__main__":
    run_full_eda()
