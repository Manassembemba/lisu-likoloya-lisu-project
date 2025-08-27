import streamlit as st
import random
from faker import Faker
from supabase import create_client, Client
import time
import auth  # Importation du module d'authentification

# --- INITIALISATION ---
fake = Faker('fr_FR')
random.seed(int(time.time()))

# Utilisation du client supabase déjà initialisé dans auth.py
supabase = auth.supabase_client

# --- CONFIGURATION ---
VILLES = {
    "Kinshasa": {"pays": "RDC", "pop": 15000000, "coords": (-4.325, 15.322)},
    "Lubumbashi": {"pays": "RDC", "pop": 2500000, "coords": (-11.66, 27.47)},
    "Goma": {"pays": "RDC", "pop": 1000000, "coords": (-1.674, 29.228)},
    "Paris": {"pays": "France", "pop": 2141000, "coords": (48.8566, 2.3522)},
    "New York": {"pays": "USA", "pop": 8399000, "coords": (40.7128, -74.0060)},
    "Dubai": {"pays": "UAE", "pop": 3137000, "coords": (25.2048, 55.2708)}
}
CATEGORIES = ["grocery_pos", "health_fitness", "shopping_pos", "shopping_net", "entertainment", "food_dining", "misc_pos", "gas_transport", "travel"]
METIERS = ["Enseignant", "Medecin", "Commercant", "Ingenieur", "Fonctionnaire", "Artiste"]

# --- LOGIQUE DE GENERATION (adaptée pour Streamlit) ---
def inserer_villes(status_placeholder):
    status_placeholder.info("Étape 1/3 : Vérification et insertion des villes...")
    city_map = {}
    db_cities = supabase.table('city').select('id_city, city_name').execute().data
    existing_cities = {c['city_name']: c['id_city'] for c in db_cities}

    for name, data in VILLES.items():
        if name not in existing_cities:
            inserted = supabase.table('city').insert({
                'city_name': name, 'country': data['pays'], 'population': data['pop'],
                'latitude': data['coords'][0], 'longitude': data['coords'][1]
            }).execute().data
            city_map[name] = inserted[0]['id_city']
        else:
            city_map[name] = existing_cities[name]
    return city_map

def creer_et_inserer_clients(n_clients, city_map, status_placeholder, progress_bar):
    status_placeholder.info(f"Étape 2/3 : Création et insertion de {n_clients} clients...")
    client_map = {}
    for i in range(n_clients):
        ville_domicile = random.choices(list(VILLES.keys()), weights=[0.5, 0.2, 0.1, 0.05, 0.1, 0.05], k=1)[0]
        client_data = {
            "cc_num": int("45" + str(random.randint(10**13, 10**14-1))),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%Y-%m-%d"),
            "job": random.choice(METIERS)
        }
        inserted = supabase.table('client').insert(client_data).execute().data
        client_id = inserted[0]['id_client']
        client_map[i] = {
            'db_id': client_id, 'ville_domicile': ville_domicile,
            'montant_moyen_depense': round(random.lognormvariate(3.5, 0.5) * 10, 2)
        }
        progress_bar.progress((i + 1) / n_clients)
    return client_map

def generer_et_inserer_transactions(n_trans, client_map, city_map, fraud_rate, status_placeholder, progress_bar):
    status_placeholder.info(f"Étape 3/3 : Création et insertion de {n_trans} transactions...")
    merchant_map = {}
    for i in range(n_trans):
        client_ref = random.choice(list(client_map.values()))
        is_fraud = random.random() < fraud_rate
        heure_transaction = random.randint(0, 23)
        ville_trans_name = random.choice([c for c in VILLES if c != client_ref['ville_domicile']]) if is_fraud else client_ref['ville_domicile']
        montant = round(client_ref["montant_moyen_depense"] * random.uniform(5, 20), 2) if is_fraud else round(max(1.0, random.normalvariate(client_ref["montant_moyen_depense"], client_ref["montant_moyen_depense"] / 3)), 2)
        
        merchant_name = fake.company()
        if merchant_name not in merchant_map:
            id_city = city_map[ville_trans_name]
            inserted_merchant = supabase.table('merchant').insert({'nom': merchant_name, 'id_city': id_city}).execute().data
            merchant_map[merchant_name] = inserted_merchant[0]['id_merchant']
        id_merchant = merchant_map[merchant_name]

        trans_data = {
            'id_client': client_ref['db_id'], 'id_merchant': id_merchant, 'amt': montant,
            'trans_time': fake.date_time_between(start_date="-3y", end_date="now").replace(hour=heure_transaction).isoformat(),
            'is_fraud': is_fraud, 'category': random.choice(CATEGORIES)
        }
        supabase.table('transaction').insert(trans_data).execute()
        progress_bar.progress((i + 1) / n_trans)

def generator_main_app():
    """ Contient l'interface principale du générateur de données."""
    st.sidebar.title(f"Connecté: {st.session_state.user.email}")
    st.sidebar.button("Se déconnecter", on_click=auth.user_logout)
    
    st.title("🛠️ Générateur de Données de Transactions")
    st.markdown("Cette interface permet d'insérer des données synthétiques directement dans la base de données Supabase.")

    if not supabase:
        st.error("La connexion à Supabase a échoué. Vérifiez les identifiants et la connexion réseau.")
        return

    with st.form("generation_form"):
        st.subheader("Paramètres de Génération")
        n_clients = st.number_input("Nombre de nouveaux clients à créer", min_value=1, max_value=1000, value=50)
        n_transactions = st.number_input("Nombre de nouvelles transactions à créer", min_value=1, max_value=10000, value=200)
        fraud_rate = st.slider("Taux de fraude", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        submitted = st.form_submit_button("🚀 Lancer la Génération et l'Insertion")

    if submitted:
        status_placeholder = st.empty()
        
        # Étape 1
        city_map = inserer_villes(status_placeholder)
        
        # Étape 2
        st.text("Création des clients...")
        progress_bar_clients = st.progress(0)
        client_map = creer_et_inserer_clients(n_clients, city_map, status_placeholder, progress_bar_clients)
        
        # Étape 3
        st.text("Création des transactions...")
        progress_bar_trans = st.progress(0)
        generer_et_inserer_transactions(n_transactions, client_map, city_map, fraud_rate, status_placeholder, progress_bar_trans)
        
        status_placeholder.success(f"Opération terminée ! {n_clients} clients et {n_transactions} ont été insérés.")

# --- ROUTEUR PRINCIPAL ---
st.set_page_config(page_title="Générateur de Données", layout="centered")

if 'user' not in st.session_state:
    auth.show_login_form()
else:
    generator_main_app()