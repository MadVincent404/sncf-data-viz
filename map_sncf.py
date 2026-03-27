import base64
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap, Fullscreen
import branca.colormap as cm
import geopandas as gpd
import json
from shapely.geometry import shape
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_lignes_ferrees() -> gpd.GeoDataFrame:
    """Charge et transforme le CSV des voies ferrées en données géospatiales."""
    print("Chargement des voies ferrées...")
    try:
        df_lignes = pd.read_csv("data/fichier-de-formes-des-voies-du-reseau-ferre-national.csv", sep=";")
        
        # Fonction pour transformer le texte JSON en géométrie Shapely
        def parse_geom(geom_str):
            if pd.isna(geom_str): 
                return None
            try:
                return shape(json.loads(geom_str))
            except Exception:
                return None

        df_lignes['geometry'] = df_lignes['Geo Shape'].apply(parse_geom)
        
        # Création du GeoDataFrame officiel
        gdf = gpd.GeoDataFrame(df_lignes, geometry='geometry')
        gdf = gdf.dropna(subset=['geometry'])
        gdf.set_crs("EPSG:4326", inplace=True)
        
        print(f"{len(gdf)} tronçons de voies chargés.")
        return gdf
    except FileNotFoundError:
        print("Fichier des voies ferrées introuvable.")
        return None

def load_and_merge() -> tuple[pd.DataFrame, dict]:
    """Fusionne gares (GPS) + fréquentation pour la carte."""
    print("Chargement des gares et fréquentations...")
    gares = pd.read_csv("data/gares-de-voyageurs.csv", sep=";")
    gares[["lat", "lon"]] = gares["Position géographique"].str.split(",", expand=True).astype(float)
    gares.rename(columns={"Nom": "gare", "Trigramme": "trigramme",
                            "Code(s) UIC": "uic_raw"}, inplace=True)
    
    gares["uic"] = gares["uic_raw"].astype(str).str.split("|").str[0].str.strip()
    gares["uic"] = pd.to_numeric(gares["uic"], errors="coerce")

    freq = pd.read_csv("data/frequentation-gares.csv", sep=";")
    freq.rename(columns={"Nom de la gare": "gare", "Code UIC": "uic",
                          "Direction Régionale Gares": "drg",
                          "Segmentation Marketing": "seg_mkt"}, inplace=True)
    year_cols = {y: f"Total Voyageurs {y}" for y in range(2015, 2025)
                 if f"Total Voyageurs {y}" in freq.columns}

    df = gares.merge(freq[["uic", "drg", "seg_mkt"] + list(year_cols.values())],
                     on="uic", how="left")

    for col in year_cols.values():
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    df["voyageurs_2024"] = df.get("Total Voyageurs 2024", pd.Series(0, index=df.index)).fillna(0)
    df["voyageurs_2019"] = df.get("Total Voyageurs 2019", pd.Series(0, index=df.index)).fillna(0)
    df["delta_covid_pct"] = np.where(
        df["voyageurs_2019"] > 0,
        (df["voyageurs_2024"] - df["voyageurs_2019"]) / df["voyageurs_2019"] * 100,
        np.nan
    )

    df = df[(df.lat.between(41, 52)) & (df.lon.between(-5.5, 10))].copy()
    df = df.dropna(subset=["lat", "lon"])
    print(f"{len(df)} gares géolocalisées chargées.")
    return df, year_cols


def get_popup_html(row) -> str:
    """Génère un popup HTML riche pour chaque gare."""
    v24 = row.get("Total Voyageurs 2024", 0) or 0
    v23 = row.get("Total Voyageurs 2023", 0) or 0
    v19 = row.get("Total Voyageurs 2019", 0) or 0
    delta = row.get("delta_covid_pct", None)
    delta_str = f"{delta:+.1f}%" if pd.notna(delta) else "N/A"
    delta_color = "#27AE60" if pd.notna(delta) and delta >= 0 else "#C0272D"

    seg = row.get("seg_mkt", "—") or "—"
    drg = row.get("drg", "—") or "—"

    return f"""
    <div style="font-family:Arial,sans-serif;font-size:12px;min-width:200px;max-width:240px">
      <div style="background:#C0272D;color:white;padding:6px 10px;border-radius:4px 4px 0 0;
                  font-weight:bold;font-size:13px">{row['gare']}</div>
      <div style="padding:8px 10px;background:#fff;border:1px solid #ddd;border-top:none;border-radius:0 0 4px 4px">
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="color:#888;padding:2px 0">Trigramme</td>
              <td style="font-weight:bold;text-align:right">{row.get('trigramme','—')}</td></tr>
          <tr><td style="color:#888;padding:2px 0">Segment</td>
              <td style="text-align:right">{seg}</td></tr>
          <tr><td style="color:#888;padding:2px 0">Direction</td>
              <td style="font-size:10px;text-align:right">{str(drg)[:25]}</td></tr>
          <tr><td colspan=2 style="padding:6px 0 2px 0;font-weight:bold;color:#333">Fréquentation</td></tr>
          <tr><td style="color:#888">2024</td>
              <td style="text-align:right;font-weight:bold">{v24:,.0f}</td></tr>
          <tr><td style="color:#888">2023</td>
              <td style="text-align:right">{v23:,.0f}</td></tr>
          <tr><td style="color:#888">2019 (pré-COVID)</td>
              <td style="text-align:right">{v19:,.0f}</td></tr>
          <tr><td style="color:#888">Δ vs 2019</td>
              <td style="text-align:right;color:{delta_color};font-weight:bold">{delta_str}</td></tr>
        </table>
      </div>
    </div>
    """


# ─────────────────────────────────────────────
# CONSTRUCTION DE LA CARTE
# ─────────────────────────────────────────────

def build_map(df: pd.DataFrame, year_cols: dict, gdf_lignes: gpd.GeoDataFrame) -> folium.Map:
    """Construit la carte Folium multi-couches."""

    m = folium.Map(
        location=[46.8, 2.3],
        zoom_start=6,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, zoom_level_offset=-6).add_to(m)

    # ── Colormap fréquentation
    max_v = df["voyageurs_2024"].quantile(0.98) 
    colormap = cm.LinearColormap(
        colors=["#FFFFCC", "#FEB24C", "#FC4E2A", "#BD0026", "#800026"],
        vmin=0, vmax=max_v,
        caption="Voyageurs 2024"
    )
    colormap.add_to(m)

    # ── NOUVELLE COUCHE : Tracé des voies ferrées (ajoutée en premier pour être en dessous)
    if gdf_lignes is not None:
        layer_lignes = folium.FeatureGroup(name="Réseau Ferré National", show=True)
        folium.GeoJson(
            gdf_lignes,
            style_function=lambda feature: {
                'color': '#C0272D',
                'weight': 1.2,
                'opacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['LIGNE', 'NOM_VOIE'], 
                aliases=['Ligne :', 'Voie :']
            )
        ).add_to(layer_lignes)
        layer_lignes.add_to(m)

    # ── Couche : HeatMap
    heat_data = df[df["voyageurs_2024"] > 0][["lat", "lon", "voyageurs_2024"]].values.tolist()
    HeatMap(
        heat_data,
        name="Heatmap fréquentation",
        min_opacity=0.3,
        max_val=float(max_v),
        radius=18, blur=22,
        gradient={"0.3": "#ffffb2", "0.6": "#fd8d3c", "0.85": "#f03b20", "1.0": "#bd0026"},
    ).add_to(m)

    # ── Couche : Cercles proportionnels (top gares)
    layer_circles = folium.FeatureGroup(name="Cercles proportionnels", show=True)
    top = df.nlargest(300, "voyageurs_2024")
    for _, row in top.iterrows():
        v = row["voyageurs_2024"]
        radius = max(3, min(30, np.log1p(v) * 1.4))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=colormap(min(v, max_v)),
            fill=True, fill_color=colormap(min(v, max_v)),
            fill_opacity=0.7, weight=0.5,
            popup=folium.Popup(get_popup_html(row), max_width=260),
            tooltip=f"<b>{row['gare']}</b><br>{v:,.0f} voy. (2024)",
        ).add_to(layer_circles)
    layer_circles.add_to(m)

    # ── Couche : Cluster toutes gares
    layer_cluster = folium.FeatureGroup(name="Toutes les gares (cluster)", show=False)
    cluster = MarkerCluster(options={"spiderfyOnMaxZoom": True, "maxClusterRadius": 40})
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(color="red", icon="train", prefix="fa"),
            popup=folium.Popup(get_popup_html(row), max_width=260),
            tooltip=row["gare"],
        ).add_to(cluster)
    cluster.add_to(layer_cluster)
    layer_cluster.add_to(m)

    # ── Couche : Delta COVID
    layer_covid = folium.FeatureGroup(name="Rebond COVID (vs 2019)", show=False)
    df_delta = df.dropna(subset=["delta_covid_pct"])
    for _, row in df_delta[df_delta["voyageurs_2019"] > 5000].iterrows():
        d = row["delta_covid_pct"]
        color = "#27AE60" if d >= 0 else "#C0272D"
        radius = max(3, min(18, abs(d) / 10))
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.65, weight=0.5,
            popup=folium.Popup(get_popup_html(row), max_width=260),
            tooltip=f"{row['gare']}: {d:+.1f}% vs 2019",
        ).add_to(layer_covid)
    layer_covid.add_to(m)

    # ── Couche : Gares A/B (grandes gares)
    layer_ab = folium.FeatureGroup(name="Grandes gares", show=False)

    # On garde ton Top 50 des gares les plus fréquentées
    grandes = df.nlargest(50, "voyageurs_2024")

    for i, (_, row) in enumerate(grandes.iterrows()):
        
        icon = folium.Icon(color="darkred", icon="train", prefix="fa")
        
        marker = folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=icon,
            popup=folium.Popup(get_popup_html(row), max_width=260)
        )
        
        # Si on est dans le Top 15, on affiche le label en permanence
        if i < 15:
            # Astuce CSS : fond transparent, pas de bordure, et un contour blanc autour des lettres
            label_style = (
                "font-size: 11px; font-weight: bold; color: #4A4A4A; "
                "background: transparent; border: none; box-shadow: none; "
                "text-shadow: -1.5px -1.5px 0 #fff, 1.5px -1.5px 0 #fff, -1.5px 1.5px 0 #fff, 1.5px 1.5px 0 #fff;"
            )
            
            folium.Tooltip(
                row['gare'],
                permanent=True,
                direction='right',
                style=label_style
            ).add_to(marker)
            
        # Si on est entre 16 et 50, on l'affiche seulement au survol pour éviter la surcharge
        else:
            folium.Tooltip(f"<b>{row['gare']}</b><br>{row['voyageurs_2024']:,.0f} voy. (2024)").add_to(marker)

        marker.add_to(layer_ab)

    layer_ab.add_to(m)
    # ── Titre carte
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                z-index:1000;background:rgba(255,255,255,0.92);
                border-left:4px solid #C0272D;padding:8px 18px;
                border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,0.15);
                font-family:Arial,sans-serif">
        <span style="font-size:15px;font-weight:bold;color:#C0272D">SNCF</span>
        <span style="font-size:13px;color:#4A4A4A;margin-left:8px">
            Carte du Réseau & Fréquentation Gares
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


def main():
    print("=" * 60)
    print("  MAP SNCF — Construction de la carte SIG")
    print("=" * 60)
    
    # 1. Charger les voies
    gdf_lignes = load_lignes_ferrees()
    
    # 2. Charger les gares
    df, year_cols = load_and_merge()
    
    # 3. Construire la carte avec tout
    m = build_map(df, year_cols, gdf_lignes)

    out = OUTPUT_DIR / "map_sncf_interactive.html"
    m.save(str(out))
    
    print(f"\nCarte sauvegardée → {out}")
    print("   Ouvre le fichier dans ton navigateur pour la visualiser.")


if __name__ == "__main__":
    main()