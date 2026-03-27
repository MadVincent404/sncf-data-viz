# SNCF Data Explorer

> Petite analyse exploratoire complète du réseau ferroviaire français · Carte SIG interactive · API temps réel

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Vue d'ensemble

Ce projet analyse **5 datasets SNCF Open Data** (2015–2024) pour répondre à des questions métier claires :
- Quelles gares ont le plus souffert du COVID et lesquelles ont rebondi ?
- À quelles heures le réseau Transilien est-il saturé ?
- Quelles liaisons TGV sont les moins ponctuelles, et pourquoi ?
- Comment fusionner données historiques, cartographie du réseau ferré et API temps réel ?

## Datasets

| Fichier | Description | Lignes |
|---------|-------------|--------|
| `frequentation-gares.csv` | Voyageurs par gare 2015–2024 | 3 021 |
| `comptage-voyageurs-trains-transilien.csv` | Montages par tranche horaire | 7 328 |
| `regularite-mensuelle-tgv-aqst.csv` | Ponctualité TGV mensuelle | 11 834 |
| `gares-de-voyageurs.csv` | Référentiel gares + coordonnées GPS | 2 782 |
| `fichier-de-formes-des-voies-du-reseau-ferre-national.csv` | Tracé géométrique des voies ferrées | > 30 000 |

## Installation & lancement

```bash
# 1. Clone
git clone [https://github.com/ton-user/sncf-data-explorer](https://github.com/ton-user/sncf-data-explorer)
cd sncf-data-explorer

# 2. Environnement
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Dépendances
pip install -r requirements.txt

# 4. Données (place les CSV dans data/)
mkdir data && cp *.csv data/

# 5. Génère les graphiques EDA
python eda_sncf.py

# 6. Génère la carte interactive
python map_sncf.py

# 7. Lance le dashboard
streamlit run app.py

## Visualisations produites

| Module | Graphique | Description |
|--------|-----------|-------------|
| `eda_sncf.py` | `00_dashboard_synthese.png` | KPIs + 6 mini-graphes |
| | `01_data_quality.png` | Complétude des données |
| | `02_frequentation_nationale.png` | Évolution 2015–2024 avec COVID |
| | `03_top_gares.png` | Top 20 + delta vs 2019 |
| | `04_transilien_lignes.png` | Trafic + heatmap horaire |
| | `05_transilien_jours.png` | Semaine vs week-end |
| | `06_tgv_regularite.png` | Ponctualité + causes |
| | `07_correlation_tgv.png` | Matrice de corrélation |
| `map_sncf.py` | `map_sncf_interactive.html` | Carte Folium multi-couches |

## Carte SIG — couches disponibles

- **Heatmap** fréquentation (intensité par zone)
- **Cercles proportionnels** Top 300 gares avec popup enrichi
- **Cluster** toutes les gares (2 782)
- **Delta COVID** : rebond vs pré-COVID 2019
- **Labels** grandes gares

## Architecture

```
sncf-data-explorer/
├── app.py              # Dashboard Streamlit multi-pages
├── eda_sncf.py         # EDA complet (matplotlib + seaborn)
├── map_sncf.py         # Carte SIG (Folium)
├── api_sncf.py         # Client API SNCF (Navitia)
├── data/               # CSV sources
├── outputs/
│   ├── eda/            # Graphiques PNG
│   └── map_sncf_interactive.html
├── requirements.txt
└── .env                # SNCF_API_TOKEN
```

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
folium>=0.14
branca>=0.6
streamlit>=1.28
streamlit-folium>=0.15
requests>=2.31
python-dotenv>=1.0
geopandas>=0.14
```

## 💡 Pistes d'amélioration

- [ ] Modèle ML de prédiction de retard (features : heure, ligne, météo API)
- [ ] Intégration météo Open-Meteo pour croiser retards + météo
- [ ] Déploiement automatique via GitHub Actions → Streamlit Cloud
- [ ] Tests unitaires des modules EDA et API
- [ ] Utilisation de l'API SNCF.
---

*Données source : [SNCF Open Data](https://data.sncf.com) · Licence ODbL*
