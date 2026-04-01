from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os

URL = os.getenv("STREAMLIT_URL", "https://sncf-data-viz-vincent.streamlit.app/")

def ping_app():
    print(f"Démarrage du ping vers {URL}...")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless=new") # Nouveau mode headless
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = None
    try:
        # Initialisation du navigateur
        driver = webdriver.Chrome(options=chrome_options)
        
        # Visite de la page
        driver.get(URL)
        
        # PAUSE CRUCIALE : On attend 15 secondes pour laisser le temps 
        # au "Zzzz" de disparaître et au code Python de l'appli de démarrer.
        print("Page chargée. Attente de l'exécution du JavaScript...")
        time.sleep(15)
        
        print("Ping terminé avec succès !")
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    ping_app()