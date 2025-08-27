import streamlit as st
from supabase import create_client, Client
import pandas as pd

# --- Initialisation de la connexion à Supabase ---
@st.cache_resource
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    try:
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Erreur de connexion à Supabase : {e}")
        return None

supabase_client = init_connection()

# --- Fonctions d'authentification ---
def show_login_form():
    st.title("🛡️ Lisu Likolo ya Lisu - Authentification")
    st.markdown("Veuillez vous connecter pour accéder au tableau de bord.")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Se connecter"):
            if email and password:
                try:
                    user_session = supabase_client.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = user_session.user
                    st.rerun()
                except Exception as e:
                    st.error(f"L'authentification a échoué. Vérifiez vos identifiants.")
            else:
                st.warning("Veuillez saisir votre email et votre mot de passe.")

def user_logout():
    if 'user' in st.session_state:
        del st.session_state['user']
    st.success("Vous avez été déconnecté avec succès.")
    st.rerun()

# --- Fonctions de données ---
def fetch_data_from_view():
    """Récupère uniquement les transactions qui n'ont pas encore été analysées."""
    try:
        # Appel de la fonction RPC pour obtenir les transactions non analysées
        response = supabase_client.rpc('get_unanalyzed_transactions', {}).execute()
        df = pd.DataFrame(response.data)
        if not df.empty and 'trans_date_trans_time' in df.columns:
            dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['hour'] = dt.dt.hour
            df['day_of_week'] = dt.dt.dayofweek
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération des transactions non analysées : {e}")
        return pd.DataFrame()

def save_frauds_to_db(results_df, prob_threshold):
    """Sauvegarde les transactions dont le score IA dépasse le seuil."""
    frauds = results_df[results_df['probability_fraud'] > prob_threshold]
    if frauds.empty:
        st.toast("Aucun nouveau cas de fraude à sauvegarder (selon le score IA).")
        return

    records_to_upsert = []
    for _, row in frauds.iterrows():
        records_to_upsert.append({
            "id_trans": row['id_trans'],
            "score": row['probability_fraud'],
            "justification": row['reason']
        })
    try:
        supabase_client.table('fraude').upsert(records_to_upsert).execute()
        st.toast(f"{len(records_to_upsert)} cas de fraude sauvegardés dans la base de données.")
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde des fraudes : {e}")

@st.cache_data(show_spinner="Chargement des cas de fraude...")
def fetch_all_frauds_details():
    try:
        response = supabase_client.rpc('get_all_fraud_cases', {}).execute()
        df = pd.DataFrame(response.data)
        if not df.empty and 'trans_date_trans_time' in df.columns:
            dt = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
            df['hour'] = dt.dt.hour
            df['day_of_week'] = dt.dt.dayofweek
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des cas de fraude : {e}")
        return pd.DataFrame()

def update_fraud_justification(id_trans, new_justification):
    try:
        supabase_client.table('fraude').update({'justification': new_justification}).eq('id_trans', id_trans).execute()
        st.success("L'explication a été sauvegardée !")
    except Exception as e:
        st.error(f"Erreur lors de la mise à jour de l'explication : {e}")