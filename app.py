import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import requests
import json
import unicodedata
import warnings
import auth  # Importation du module d'authentification

warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

# ===============================
# --- CONFIGURATION DE LA PAGE ---
# ===============================
st.set_page_config(
    page_title="Lisu Likolo ya Lisu - Détection de Fraude Bancaire",
    page_icon="LOGO.jpg",
    layout="wide"
)

# ===============================
# --- STYLES CSS ---
# ===============================
st.markdown(r'''
<style>
:root {
    --primary-color: #2E86C1;
    --accent-color: #F39C12;
}
h1, h2, h3 { color: var(--primary-color); font-family: 'Segoe UI', sans-serif; }
div.stButton > button:first-child { background-color: var(--primary-color); color: white; border-radius: 8px; height: 3em; font-weight: bold; transition: background-color 0.3s ease; }
div.stButton > button:first-child:hover { background-color: #1F618D; }
.robot-icon { font-size: 2em; cursor: pointer; color: var(--accent-color); }
</style>
''', unsafe_allow_html=True)

# ===============================
# --- FONCTIONS UTILITAIRES ---
# ===============================
@st.cache_resource
def load_artefacts():
    artefacts = {}
    try:
        artefacts["Random Forest"] = joblib.load("pipeline_rf.joblib")
        artefacts["XGBoost"] = joblib.load("pipeline_xgb.joblib")
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
    return artefacts

def prepare_features(df, model):
    df_prepared = df.copy()
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
        for col in expected_cols:
            if col not in df_prepared.columns:
                df_prepared[col] = 0
        df_prepared = df_prepared.reindex(columns=expected_cols, fill_value=0)
    return df_prepared

def analyze_data(df, model_pipeline):
    model = model_pipeline['model'] if isinstance(model_pipeline, dict) and 'model' in model_pipeline else model_pipeline
    df_prepared = prepare_features(df.copy(), model) # Utilise une copie pour éviter de modifier l'original
    try:
        probs = model.predict_proba(df_prepared)[:, 1]
    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
        return df
    results = df.copy()
    results['probability_fraud'] = probs
    return results

def clean_response_text(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

def paginate_dataframe(df, page_size=50, key_prefix="pagination"):
    if f"{key_prefix}_page_number" not in st.session_state:
        st.session_state[f"{key_prefix}_page_number"] = 0
    total_pages = (len(df) - 1) // page_size + 1 if len(df) > 0 else 1
    start = st.session_state[f"{key_prefix}_page_number"] * page_size
    end = start + page_size
    st.dataframe(df.iloc[start:end])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅️ Page précédente", key=f"{key_prefix}_prev"):
            st.session_state[f"{key_prefix}_page_number"] = max(st.session_state[f"{key_prefix}_page_number"] - 1, 0)
    with col2:
        st.write(f"Page {st.session_state.get(f'{key_prefix}_page_number', 0) + 1} / {total_pages}")
    with col3:
        if st.button("➡️ Page suivante", key=f"{key_prefix}_next"):
            st.session_state[f"{key_prefix}_page_number"] = min(st.session_state[f"{key_prefix}_page_number"] + 1, total_pages - 1)

# ===============================
# --- RENDU DES COMPOSANTS UI ---
# ===============================
def render_dashboard(results_df, model_pipeline, selected_model_name):
    st.subheader("📊 Résultats de l'analyse en direct")
    col1, col2, col3 = st.columns(3)
    fraud_filter = col1.selectbox("Type de transaction", ["Toutes", "Transactions suspectes", "Transactions normales"], key="dash_filter")
    min_amount = col2.number_input("Montant minimum ($)", min_value=0.0, value=0.0, step=10.0, key="dash_min_amount")
    category_filter = col3.multiselect("Catégories", options=results_df['category'].unique(), default=results_df['category'].unique(), key="dash_cat_filter")

    filtered_df = results_df.copy()
    if fraud_filter == "Transactions suspectes":
        filtered_df = filtered_df[filtered_df['is_flagged'] == True]
    elif fraud_filter == "Transactions normales":
        filtered_df = filtered_df[filtered_df['is_flagged'] == False]
    filtered_df = filtered_df[filtered_df['amt'] >= min_amount]
    filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]

    st.info(f"🔎 {len(filtered_df):,} transactions correspondent aux critères")
    display_cols = ['id_trans', 'trans_date_trans_time', 'amt', 'category', 'merchant', 'city', 'probability_fraud', 'is_flagged', 'reason']
    paginate_dataframe(filtered_df[display_cols], key_prefix="dash_pagination")

    fraud_df = results_df[results_df['is_flagged'] == True]
    if not fraud_df.empty:
        st.markdown("### 📈 Statistiques des transactions suspectes")
        cols = st.columns(4)
        cols[0].metric("Total Suspectes", f"{len(fraud_df):,}")
        cols[1].metric("Taux de Suspicion", f"{len(fraud_df)/len(results_df):.2%}" if len(results_df) > 0 else "N/A")
        cols[2].metric("Montant Moyen Suspect", f"{fraud_df['amt'].mean():,.2f} $")
        cols[3].metric("Montant Total Suspect", f"{fraud_df['amt'].sum():,.2f} $")

        st.markdown("### 🎨 Analyses Visuelles des Cas Suspects")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Montants", "Heures", "Catégories", "Jours", "Carte"])
        with tab1:
            fig = px.histogram(fraud_df.sample(min(len(fraud_df), 50000)), x='amt', title='Distribution des montants suspects')
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.bar_chart(fraud_df['hour'].value_counts().sort_index())
        with tab3:
            st.bar_chart(fraud_df['category'].value_counts())
        with tab4:
            st.bar_chart(fraud_df['day_of_week'].value_counts().sort_index())
        with tab5:
            if 'merch_lat' in fraud_df.columns and 'merch_long' in fraud_df.columns:
                map_data = fraud_df[['merch_lat', 'merch_long']].rename(columns={'merch_lat':'lat','merch_long':'lon'})
                st.map(map_data.dropna().sample(min(len(map_data.dropna()), 5000)))

        if selected_model_name == "XGBoost":
            st.markdown("### 🕵️ Analyse SHAP d'une transaction suspecte")
            try:
                model = model_pipeline['model']
                final_estimator = model.steps[-1][1]
                explainer = shap.TreeExplainer(final_estimator)
                options_shap = {f"#{i} – Montant: {r['amt']:.2f} – Cat: {r['category']}": i for i, r in fraud_df.iterrows()}
                if options_shap:
                    selected_label = st.selectbox("Choisissez une transaction à expliquer avec SHAP :", list(options_shap.keys()))
                    idx = options_shap[selected_label]
                    transaction = fraud_df.loc[[idx]]
                    prepared_data = prepare_features(transaction, model)
                    shap_values = explainer(prepared_data)
                    st.markdown("#### Impact des facteurs sur la prédiction :")
                    fig, ax = plt.subplots()
                    shap.plots.bar(shap_values[0], show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP non disponible pour ce modèle : {e}")

def render_case_management(selected_model_name):
    st.subheader("🗃️ Gestion des Cas de Fraude Enregistrés")
    if st.button("🔄 Rafraîchir les cas"):
        auth.fetch_all_frauds_details.clear()
        if 'lisu_selected_case_id' in st.session_state:
            del st.session_state.lisu_selected_case_id
        st.rerun()

    all_frauds_df = auth.fetch_all_frauds_details()
    if all_frauds_df.empty:
        st.info("Aucun cas de fraude n'a été enregistré.")
        return

    display_df = all_frauds_df.copy()
    display_df.insert(0, "Sélectionner", False)

    if 'lisu_selected_case_id' in st.session_state:
        selected_id = st.session_state.lisu_selected_case_id
        if not display_df[display_df['id_trans'] == selected_id].empty:
            display_df.loc[display_df['id_trans'] == selected_id, 'Sélectionner'] = True

    st.info("Cochez la case de la ligne que vous souhaitez analyser.")
    edited_df = st.data_editor(
        display_df,
        key="fraud_case_editor",
        hide_index=True,
        use_container_width=True,
        disabled=[col for col in all_frauds_df.columns if col != "Sélectionner"]
    )

    selected_rows = edited_df[edited_df["Sélectionner"]]

    if not selected_rows.empty:
        selected_id = selected_rows.iloc[-1]['id_trans']
        if st.session_state.get('lisu_selected_case_id') != selected_id:
            st.session_state.lisu_selected_case_id = selected_id
            st.session_state.n8n_response = None
            st.rerun()
    elif 'lisu_selected_case_id' in st.session_state:
        del st.session_state.lisu_selected_case_id
        st.rerun()

    st.markdown("---" )
    st.markdown("### 🤖 Agent AI Mr.LISU - Analyse experte d'un cas")

    if 'lisu_selected_case_id' in st.session_state:
        selected_id = st.session_state.lisu_selected_case_id
        selected_transaction = all_frauds_df[all_frauds_df['id_trans'] == selected_id]
        
        st.success(f"✅ Transaction #{selected_id} sélectionnée.")
        st.write(selected_transaction)

        user_msg = st.text_area("Votre question pour Mr.LISU :", key="n8n_question")
        if st.button("📨 Envoyer à Mr.LISU", key="n8n_send"):
            if user_msg.strip():
                transaction_data = selected_transaction.iloc[0].to_dict()
                payload = {
                    "src": {
                        "question": str(user_msg).strip(),
                        "model": str(selected_model_name),
                        "id_trans": int(transaction_data.get("id_trans", 0)),
                        "id_client": int(transaction_data.get("id_client", 0)),
                        "trans_datetime": str(transaction_data.get("trans_date_trans_time")),
                        "card_number": int(transaction_data.get("cc_num", 0)),
                        "merchant": str(transaction_data.get("merchant", "")),
                        "category": str(transaction_data.get("category", "")),
                        "amount": float(transaction_data.get("amt", 0.0)),
                        "gender": str(transaction_data.get("gender", "")),
                        "city": str(transaction_data.get("city", "")),
                        "state": str(transaction_data.get("state", "")),
                        "zip": str(transaction_data.get("zip", "")),
                        "city_population": str(transaction_data.get("city_pop", "")),
                        "job": str(transaction_data.get("job", "")),
                        "dob": str(transaction_data.get("dob", "")),
                        "merchant_lat": float(transaction_data.get("merch_lat", 0.0)),
                        "merchant_long": float(transaction_data.get("merch_long", 0.0)),
                        "hour": int(transaction_data.get("hour", 0)),
                        "day_of_week": int(transaction_data.get("day_of_week", 0)),
                        "is_fraud_predicted": bool(transaction_data.get("is_fraud", False)),
                        "probability_fraud": float(transaction_data.get("score", 0.0))
                    }
                }
                n8n_url = "https://n8n-dw1u.onrender.com/webhook/fraudeExplain"
                try:
                    with st.spinner("Envoi de la requête à l'agent Mr.LISU..."):
                        resp = requests.post(n8n_url, json=payload, timeout=30)
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            st.session_state.n8n_response = data
                        except requests.exceptions.JSONDecodeError:
                            st.session_state.n8n_response = {"reponse_brute": resp.text}
                        st.session_state.n8n_trans_id = selected_transaction.iloc[0]['id_trans']
                        st.rerun()
                    else:
                        st.error(f"Erreur du service N8N (Code: {resp.status_code}): {resp.text}")
                except requests.exceptions.Timeout:
                    st.error("La requête a dépassé le temps limite.")
                except requests.exceptions.ConnectionError:
                    st.error("Erreur de connexion à N8N.")
                except Exception as e:
                    st.error(f"Une erreur inattendue est survenue : {type(e).__name__} - {e}")
            else:
                st.warning("Veuillez saisir une question.")

       # 🔗 Vérifie que la réponse correspond bien à la transaction actuelle
        if 'n8n_response' in st.session_state and 'n8n_trans_id' in st.session_state:
            if st.session_state.n8n_trans_id == selected_transaction.iloc[0]['id_trans']:
                st.markdown("**Réponse de Mr.LISU :**")
                response_data = st.session_state.n8n_response
                if isinstance(response_data, dict) and 'output' in response_data:
                    cleaned_text = clean_response_text(response_data['output'])
                    st.markdown(cleaned_text, unsafe_allow_html=True)
                else:
                    st.warning("Réponse du système (possible erreur) :")
                    st.json(response_data)
            else:
                # ❌ La réponse ne correspond pas à cette transaction
                del st.session_state.n8n_response
                del st.session_state.n8n_trans_id
                st.rerun()
            if st.button("💾 Sauvegarder cette explication", key="n8n_save"):
                explanation_str = json.dumps(st.session_state.n8n_response, indent=2, ensure_ascii=False)
                auth.update_fraud_justification(st.session_state.n8n_trans_id, explanation_str)
                del st.session_state.n8n_response
                del st.session_state.n8n_trans_id
                st.rerun()
    else:
        st.warning("Veuillez cocher la case d'une transaction pour l'analyser.")

# ===============================
# --- APPLICATION PRINCIPALE ---
# ===============================
def main_app():
    st.sidebar.title(f"Bienvenue, {st.session_state.user.email}")
    st.sidebar.button("Se déconnecter", on_click=auth.user_logout)
    st.sidebar.markdown("---")

    selected_model_name = st.sidebar.radio("Modèle IA :", list(artefacts.keys()), key="model_select")
    model_pipeline = artefacts[selected_model_name]

    with st.sidebar.expander("⚙️ Paramètres de Détection", expanded=True):
        st.session_state.prob_threshold = st.slider('Seuil de probabilité', 0.0, 1.0, st.session_state.get('prob_threshold', 0.75), 0.05)
        st.session_state.high_value_rule = st.number_input('Seuil de montant élevé ($)', 0, value=st.session_state.get('high_value_rule', 3000), step=100)

    st.title("🛡️ Lisu Likolo ya Lisu")
    tab1, tab2 = st.tabs(["Analyse en Direct", "Gestion des Cas"])

    with tab1:
        st.header("Analyse de nouvelles transactions")
        if st.button("🔄 Charger et Analyser les Transactions"):
            df = auth.fetch_data_from_view()
            if not df.empty:
                with st.spinner("Analyse IA en cours..."):
                    results_df = analyze_data(df, model_pipeline)
                    prob_threshold = st.session_state.prob_threshold
                    high_value_rule = st.session_state.high_value_rule
                    rule1 = results_df['probability_fraud'] > prob_threshold
                    rule2 = results_df['amt'] > high_value_rule
                    results_df['is_flagged'] = rule1 | rule2
                    results_df['reason'] = np.select([rule1 & rule2, rule1, rule2], ['Score & Montant Élevés', 'Score IA Élevé', 'Montant Élevé'], default='-')
                    auth.save_frauds_to_db(results_df, st.session_state.prob_threshold)
                    st.session_state.results_df = results_df
                st.success("Analyse terminée et cas de fraude sauvegardés.")
            else:
                st.warning("Aucune nouvelle transaction à analyser.")
                if 'results_df' in st.session_state:
                    del st.session_state.results_df
        
        if 'results_df' in st.session_state:
            render_dashboard(st.session_state.results_df, model_pipeline, selected_model_name)

    with tab2:
        st.header("Investigation des cas de fraude enregistrés")
        render_case_management(selected_model_name)

# ===============================
# --- ROUTEUR PRINCIPAL ---
# ===============================
if 'user' not in st.session_state:
    auth.show_login_form()
else:
    artefacts = load_artefacts()
    if not artefacts:
        st.error("Impossible de charger les modèles IA. L'application ne peut pas démarrer.")
        st.stop()
    main_app()
