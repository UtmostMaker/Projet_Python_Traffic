# app/app.py
# Script principal pour l'application web Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os
import sys

# --- Importation des Classes Personnalisées ---
# Ajoute le répertoire racine du projet au chemin Python pour trouver 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Importe depuis le dossier 'src'
try:
    from src.custom_transformers import FrequencyEncoder
except ModuleNotFoundError:
    st.error(
        "Erreur Critique: Impossible d'importer 'FrequencyEncoder' depuis "
        "'src.custom_transformers'. Vérifiez l'existence du fichier et "
        "la structure du projet."
    )
    st.stop()
except Exception as e:
    st.error(
        f"Erreur inattendue lors de l'importation "
        f"du transformateur personnalisé: {e}"
    )
    st.stop()

# --- Configuration ---
PAGE_TITLE = "Prédicteur de Volume de Trafic Inter-États"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILENAME = "traffic_prediction_RandomForest_tuned.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# --- Seuils Visuels et Nouvelle Palette de Couleurs ---
CONGESTION_THRESHOLD = 4500
HIGH_TRAFFIC_THRESHOLD = 3500
MEDIUM_TRAFFIC_THRESHOLD = 2000
LOW_TRAFFIC_THRESHOLD = 1000

# Couleurs (Vert -> Rouge)
COLOR_CONGESTION = "#dc3545"    # Rouge Vif
COLOR_HIGH = "#fd7e14"          # Orange
COLOR_MEDIUM = "#ffc107"        # Jaune
COLOR_LOW = "#20c997"           # Turquoise/Vert Clair
COLOR_VERY_LOW = "#198754"      # Vert Foncé
COLOR_TEXT_DARK = "#000000"     # Noir
COLOR_TEXT_LIGHT = "#FFFFFF"    # Blanc

st.set_page_config(
    page_title=PAGE_TITLE, layout="wide", initial_sidebar_state="expanded"
)

# --- Fonctions Auxiliaires ---

def get_peak_hour(hour):
    """Catégorise l'heure en périodes de pointe/hors pointe."""
    if 7 <= hour < 10:
        return 'morning_peak'
    elif 16 <= hour < 19:
        return 'evening_peak'
    else:
        return 'off_peak'

def get_season(month):
    """Détermine la saison en fonction du mois."""
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:  # 9, 10, 11
        return 'autumn'

def get_temp_category(temp_celsius):
    """Catégorise la température en labels prédéfinis."""
    conditions = [
        temp_celsius < 0,
        (temp_celsius >= 0) & (temp_celsius < 15),
        (temp_celsius >= 15) & (temp_celsius < 25),
        temp_celsius >= 25
    ]
    choices = ['freezing', 'cold', 'mild', 'hot']
    return np.select(conditions, choices, default='unknown')

def get_traffic_color_and_text_color(volume):
    """Retourne les couleurs de fond et de texte selon le volume."""
    if volume > CONGESTION_THRESHOLD:
        return COLOR_CONGESTION, COLOR_TEXT_LIGHT
    elif volume > HIGH_TRAFFIC_THRESHOLD:
        return COLOR_HIGH, COLOR_TEXT_LIGHT
    elif volume > MEDIUM_TRAFFIC_THRESHOLD:
        return COLOR_MEDIUM, COLOR_TEXT_DARK
    elif volume > LOW_TRAFFIC_THRESHOLD:
        return COLOR_LOW, COLOR_TEXT_DARK
    else:
        return COLOR_VERY_LOW, COLOR_TEXT_LIGHT

def get_congestion_status_html(volume):
    """Retourne le HTML pour l'état de congestion (Français)."""
    if volume > CONGESTION_THRESHOLD:
        text = "🚦 Congestion Possible"
        text_color = COLOR_TEXT_LIGHT
        bg_color = COLOR_CONGESTION
        font_weight = "bold"
    else:
        text = "✅ Trafic Probablement Fluide"
        text_color = "#198754"  # Vert foncé pour texte fluide
        bg_color = "transparent"
        font_weight = "normal"

    # Construction du style
    style = (
        f"color: {text_color}; "
        f"background-color: {bg_color}; "
        f"font-weight: {font_weight}; "
        f"padding: 5px 10px; "
        f"border-radius: 5px; "
        f"display: inline-block; "
        f"margin-top: 8px;"
    )
    return f'<h4 style="{style}">{text}</h4>'

# --- Chargement du Pipeline Modèle ---
@st.cache_resource
def load_pipeline(path):
    """Charge le pipeline sérialisé."""
    if not os.path.exists(path):
        st.error(f"Erreur : Fichier modèle introuvable : '{path}'.")
        return None
    try:
        pipeline = joblib.load(path)
        print(f"Pipeline chargé avec succès : {path}.")
        return pipeline
    except Exception as e:
        st.error(f"Erreur lors du chargement du pipeline : {e}")
        return None

pipeline = load_pipeline(MODEL_PATH)

# --- Interface Utilisateur ---
st.title(f"🚘 {PAGE_TITLE}")

st.markdown("""
Entrez les conditions ci-dessous pour estimer le volume horaire du trafic
sur l'autoroute I-94 (Minneapolis-St Paul).
*(Basé sur des données historiques 2012-2018)*
""")

# Widgets de saisie
col1, col2 = st.columns(2)
with col1:
    st.subheader("📅 Date et Heure")
    input_date = st.date_input("Sélectionner la Date", datetime.date.today())
    input_hour = st.slider("Sélectionner l'Heure (0-23)", 0, 23, 12)

    st.subheader("🗓️ Statut Jour Férié")
    possible_holidays = [
        'None', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
        'Christmas Day', 'New Years Day', 'Washingtons Birthday',
        'State Fair', 'Memorial Day', 'Independence Day', 'Labor Day',
        'Martin Luther King Jr Day'
    ]
    input_holiday = st.selectbox(
        "Statut du Jour Férié",
        possible_holidays,
        index=possible_holidays.index('None')
    )

with col2:
    st.subheader("☁️ Conditions Météo")
    # Valeurs anglaises (pour le modèle) et dictionnaire de traduction (pour l'UI)
    possible_weather_en = [
        'Clear', 'Clouds', 'Rain', 'Mist', 'Snow', 'Drizzle', 'Haze',
        'Fog', 'Thunderstorm', 'Squall', 'Smoke'
    ]
    weather_translation = {
        'Clear': 'Dégagé', 'Clouds': 'Nuageux', 'Rain': 'Pluie',
        'Mist': 'Brume', 'Snow': 'Neige', 'Drizzle': 'Bruine',
        'Haze': 'Brume sèche', 'Fog': 'Brouillard',
        'Thunderstorm': 'Orage', 'Squall': 'Rafale', 'Smoke': 'Fumée'
    }
    # Fonction pour afficher la traduction dans le selectbox
    def format_weather(option_en):
        return weather_translation.get(option_en, option_en)

    input_weather = st.selectbox(
        "Condition Météo Principale",
        options=possible_weather_en, # Options réelles en anglais
        index=possible_weather_en.index('Clear'),
        format_func=format_weather # Affiche la traduction
     )

    input_temp_celsius = st.number_input(
        "Température (°C)", -40.0, 50.0, 15.0, 0.5
    )
    input_rain = st.number_input(
        "Précipitations Pluie (mm/h)", 0.0, 100.0, 0.0, 0.1, "%.2f"
    )
    input_snow = st.number_input(
        "Précipitations Neige (mm/h)", 0.0, 5.0, 0.0, 0.1, "%.2f"
    )
    input_clouds = st.slider(
        "Couverture Nuageuse (%)", 0, 100, 50, 5
    )

# --- Logique de Prédiction ---
def prepare_single_input(target_date, target_hour, holiday, temp, rain,
                         snow, clouds, weather):
    """Prépare une ligne de données d'entrée (dictionnaire)."""
    weekday = target_date.weekday()
    month = target_date.month
    year = target_date.year
    return {
        'holiday': holiday, 'temp': temp, 'rain_1h': rain, 'snow_1h': snow,
        'clouds_all': clouds, 'weather_main': weather, 'hour': target_hour,
        'day_of_week': weekday, 'month': month, 'year': year,
        'is_weekend': 1 if weekday >= 5 else 0,
        'peak_hour': get_peak_hour(target_hour), 'season': get_season(month),
        'is_precipitating': 1 if (rain + snow) > 0 else 0,
        'temp_category': get_temp_category(temp)
    }

# Bouton principal
if st.button("📈 Prédire le Volume de Trafic", type="primary"):
    if pipeline is not None:
        try:
            # 1. Prepare les inputs pour H-1 et H+1
            prediction_inputs = []
            base_dt = datetime.datetime.combine(input_date,
                                                datetime.time(input_hour))
            timestamps = [base_dt + datetime.timedelta(hours=h) for h in [-1, 0, 1]]

            for ts in timestamps:
                single_input = prepare_single_input(
                    target_date=ts.date(), target_hour=ts.hour,
                    holiday=input_holiday, temp=input_temp_celsius,
                    rain=input_rain, snow=input_snow, clouds=input_clouds,
                    weather=input_weather # Passes les valeurs en anglais
                )
                prediction_inputs.append(single_input)

            input_data_multi_hour = pd.DataFrame(prediction_inputs)

            # 2. Prédiction
            predictions = pipeline.predict(input_data_multi_hour)

            # 3. Affiche les résultats
            st.markdown("---")
            st.subheader("Résultats de la Prédiction")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            labels = ["Heure Précédente", "Heure Sélectionnée", "Heure Suivante"]

            for i, col in enumerate([pred_col1, pred_col2, pred_col3]):
                with col:
                    pred_volume = int(round(predictions[i]))
                    pred_volume = max(0, pred_volume)
                    time_str = timestamps[i].strftime('%H:%M')
                    date_str = timestamps[i].strftime('%Y-%m-%d')

                    # Colorise la cellule  des résultats
                    bg_color, text_color = get_traffic_color_and_text_color(
                        pred_volume
                    )

                    # Construction de contenu HTML pour un affichage plus clair
                    html_content = f"""
<div style="background-color:{bg_color}; color:{text_color};
            padding: 0.8em; border-radius: 8px; text-align: center;
            margin-bottom: 10px; display: flex; flex-direction: column;
            justify-content: center; align-items:center; min-height: 125px;">
    <div style="font-size: 0.85em; font-weight: bold;
                margin-bottom: 0.3em;">
        {labels[i]}
    </div>
    <div style="font-size: 0.75em; margin-bottom: 0.5em;">
        {date_str} {time_str}
    </div>
    <div style="font-size: 1.6em; font-weight: bold; line-height: 1.1;">
        {pred_volume}
    </div>
    <div style="font-size: 0.75em;">
        véhicules/heure
    </div>
</div>
"""
                    st.markdown(html_content, unsafe_allow_html=True)

                    # Ajoute le statut de congestion
                    if i == 1:
                        congestion_html = get_congestion_status_html(pred_volume)
                        # Centre le texte
                        st.markdown(
                            f"<div style='text-align: center;'>{congestion_html}</div>",
                            unsafe_allow_html=True
                        )

            st.success("Prédictions générées avec succès !")

            with st.expander("Afficher les données d'entrée (3 heures)"):
                st.dataframe(input_data_multi_hour.astype(str))

        except ValueError as ve:
             st.error(f"Erreur de Saisie : {ve}. Vérifiez les valeurs.")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
    else:
        st.error(
            "Prédiction impossible : le pipeline modèle n'a pas pu être chargé."
        )

# --- Sidebar ---
st.sidebar.header("À Propos de ce Projet")
st.sidebar.info(
    """
    Cette application prédit le volume horaire du trafic sur l'autoroute I-94
    en utilisant un modèle Machine Learning pré-entraîné.

    **Modèle :** RandomForest Regressor (optimisé)
    **Source Données :** Kaggle Metro Interstate Traffic Volume (2012-2018)
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Développé dans le cadre d'un projet de cours Python. "
    "Les caractéristiques utilisées incluent l'heure, la date, les jours fériés, "
    "et les conditions météorologiques."
)
# GITHUB_REPO = "https://github.com/UtmostMaker/Projet_Python_Traffic"
# st.sidebar.markdown(f"[Voir le Projet sur GitHub]({GITHUB_REPO})")

