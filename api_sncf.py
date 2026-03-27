import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SNCF_TOKEN = os.getenv("SNCF_API_TOKEN", "TON_TOKEN_ICI")
BASE_URL    = "https://api.sncf.com/v1/coverage/sncf"


# ─────────────────────────────────────────────
# CLIENT DE BASE
# ─────────────────────────────────────────────

class SNCFClient:
    """Client léger pour l'API SNCF Open Data (Navitia)."""

    def __init__(self, token: str = SNCF_TOKEN):
        self.session = requests.Session()
        self.session.auth = (token, "")
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict | None:
        url = f"{BASE_URL}/{endpoint}"
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=10)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                if r.status_code == 401:
                    print("Token invalide — vérifie ton SNCF_API_TOKEN dans .env")
                    return None
                if r.status_code == 429:
                    print(f"Rate limit, attente {2**attempt}s...")
                    time.sleep(2 ** attempt)
                else:
                    print(f"HTTP {r.status_code} : {e}")
                    return None
            except requests.exceptions.ConnectionError:
                print("Pas de connexion réseau")
                return None
        return None


# ─────────────────────────────────────────────
# 1. RECHERCHE DE GARE
# ─────────────────────────────────────────────

def search_stop(client: SNCFClient, query: str) -> list[dict]:
    """Cherche une gare par nom, retourne les résultats."""
    data = client._get("places", params={"q": query, "type[]": "stop_area", "count": 5})
    if not data:
        return []
    results = []
    for place in data.get("places", []):
        results.append({
            "id": place["id"],
            "name": place["name"],
            "coord": place.get("stop_area", {}).get("coord", {}),
        })
    return results


# ─────────────────────────────────────────────
# 2. PROCHAINS PASSAGES
# ─────────────────────────────────────────────

def get_next_departures(client: SNCFClient, stop_id: str, count: int = 10) -> pd.DataFrame:
    """
    Retourne les prochains départs d'une gare.
    stop_id : ex. "stop_area:SNCF:87391003" (Paris Montparnasse)
    """
    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    data = client._get(
        f"stop_areas/{stop_id}/departures",
        params={"from_datetime": now, "count": count, "duration": 3600}
    )
    if not data:
        return pd.DataFrame()

    rows = []
    for dep in data.get("departures", []):
        info = dep.get("display_informations", {})
        stop_dt = dep.get("stop_date_time", {})
        rows.append({
            "ligne":         info.get("headsign", "—"),
            "direction":     info.get("direction", "—"),
            "reseau":        info.get("network", "—"),
            "depart_prevu":  stop_dt.get("base_departure_date_time", ""),
            "depart_reel":   stop_dt.get("departure_date_time", ""),
            "type":          info.get("commercial_mode", "—"),
            "statut":        stop_dt.get("data_freshness", "—"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Formatage des heures
    for col in ["depart_prevu", "depart_reel"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%dT%H%M%S", errors="coerce")

    df["retard_min"] = (df["depart_reel"] - df["depart_prevu"]).dt.total_seconds() / 60
    df["retard_min"] = df["retard_min"].fillna(0).astype(int)
    df["a_lheure"] = df["retard_min"] <= 5
    return df


# ─────────────────────────────────────────────
# 3. PERTURBATIONS EN COURS
# ─────────────────────────────────────────────

def get_disruptions(client: SNCFClient, stop_id: str = None) -> pd.DataFrame:
    """
    Retourne les perturbations actives.
    Si stop_id fourni, filtre sur la gare ; sinon retourne tout.
    """
    endpoint = f"stop_areas/{stop_id}/disruptions" if stop_id else "disruptions"
    data = client._get(endpoint, params={"count": 50, "current_datetime": True})
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data.get("disruptions", []):
        rows.append({
            "id":          d.get("id", ""),
            "statut":      d.get("status", ""),
            "severite":    d.get("severity", {}).get("name", ""),
            "effet":       d.get("severity", {}).get("effect", ""),
            "titre":       d.get("messages", [{}])[0].get("text", "")[:80] if d.get("messages") else "",
            "debut":       d.get("application_periods", [{}])[0].get("begin", ""),
            "fin":         d.get("application_periods", [{}])[0].get("end", ""),
            "lignes":      ", ".join(
                li.get("name", "") for li in d.get("impacted_objects", [])
                if li.get("pt_object", {}).get("embedded_type") == "line"
            )[:60],
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. ITINÉRAIRE
# ─────────────────────────────────────────────

def get_journey(client: SNCFClient, from_id: str, to_id: str,
                datetime_str: str = None) -> pd.DataFrame:
    """
    Calcule un itinéraire entre 2 gares.
    from_id, to_id : stop_area IDs SNCF
    datetime_str   : format "20250101T080000" (défaut = maintenant)
    """
    if datetime_str is None:
        datetime_str = datetime.now().strftime("%Y%m%dT%H%M%S")

    data = client._get("journeys", params={
        "from": from_id, "to": to_id,
        "datetime": datetime_str,
        "count": 3,
        "min_nb_journeys": 2,
    })
    if not data:
        return pd.DataFrame()

    rows = []
    for j in data.get("journeys", []):
        duration_min = j.get("duration", 0) // 60
        sections = j.get("sections", [])
        trains = [s for s in sections if s.get("type") == "public_transport"]
        train_info = " + ".join(
            s.get("display_informations", {}).get("headsign", "?")
            for s in trains
        )
        rows.append({
            "depart":          j.get("departure_date_time", ""),
            "arrivee":         j.get("arrival_date_time", ""),
            "duree_min":       duration_min,
            "nb_correspondances": max(0, len(trains) - 1),
            "trains":          train_info,
            "statut":          j.get("status", ""),
            "CO2_kg":          j.get("co2_emission", {}).get("value", 0),
        })

    df = pd.DataFrame(rows)
    for col in ["depart", "arrivee"]:
        df[col] = pd.to_datetime(df[col], format="%Y%m%dT%H%M%S", errors="coerce")
    return df


# ─────────────────────────────────────────────
# 5. ENRICHISSEMENT DES DONNÉES HISTORIQUES
# ─────────────────────────────────────────────

def enrich_with_realtime(df_hist: pd.DataFrame, client: SNCFClient,
                          gare_col: str = "gare",
                          n_gares: int = 10) -> pd.DataFrame:
    """
    Ajoute des infos live (nb perturbations actives) aux données historiques.
    Pratique pour le dashboard Streamlit : colonne "perturbations_live".
    """
    top_gares = df_hist.nlargest(n_gares, "voyageurs_2024")[gare_col].tolist()
    perturb_counts = {}

    for gare in top_gares:
        stops = search_stop(client, gare)
        if stops:
            disrupt = get_disruptions(client, stops[0]["id"])
            perturb_counts[gare] = len(disrupt)
        time.sleep(0.3)   # Respecte le rate limit (~3 req/s)

    df_hist["perturbations_live"] = df_hist[gare_col].map(perturb_counts)
    return df_hist


# ─────────────────────────────────────────────
# 6. DÉMO / USAGE
# ─────────────────────────────────────────────

def demo():
    """Démonstration rapide de l'API."""
    print("=" * 60)
    print("  SNCF API — Démo temps réel")
    print("=" * 60)

    client = SNCFClient()

    # Recherche Paris Montparnasse
    print("\nRecherche 'Paris Montparnasse'...")
    stops = search_stop(client, "Paris Montparnasse")
    if not stops:
        print("Aucun résultat — vérifie ton token dans .env")
        return

    for s in stops[:3]:
        print(f"   {s['name']} → {s['id']}")

    stop_id = stops[0]["id"]

    # Prochains départs
    print(f"\nProchains départs depuis {stops[0]['name']}...")
    df_dep = get_next_departures(client, stop_id, count=8)
    if not df_dep.empty:
        print(df_dep[["direction", "type", "depart_reel", "retard_min", "a_lheure"]].to_string(index=False))

    # Perturbations
    print(f"\nPerturbations actives...")
    df_dis = get_disruptions(client, stop_id)
    if df_dis.empty:
        print("   Aucune perturbation — bonne nouvelle !")
    else:
        print(df_dis[["severite", "effet", "titre", "lignes"]].head(5).to_string(index=False))

    # Statistiques live
    if not df_dep.empty:
        print("\nStatistiques en temps réel :")
        print(f"   Trains à l'heure : {df_dep['a_lheure'].sum()}/{len(df_dep)}")
        print(f"   Retard moyen     : {df_dep['retard_min'].mean():.1f} min")
        on_time_rate = df_dep["a_lheure"].mean() * 100
        print(f"   Ponctualité live : {on_time_rate:.0f}%")


if __name__ == "__main__":
    demo()
