import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models and encoders
intent_model = joblib.load('model/intent_model.pkl')
domain_model = joblib.load('model/domain_model.pkl')
intent_encoder = joblib.load('model/intent_encoder.pkl')
domain_encoder = joblib.load('model/domain_encoder.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Load from hub for now

# Title
st.title("Query Classifier 🔍")
st.markdown("Classifies user queries into intent and domain (finance or healthcare)")

# Input
user_input = st.text_area("Enter your query", "")

# Prediction
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a query.")
    else:
        # Embed and predict
        embedding = embedder.encode([user_input])[0].reshape(1, -1)

        intent_proba = intent_model.predict_proba(embedding)
        domain_proba = domain_model.predict_proba(embedding)

        intent_idx = np.argmax(intent_proba)
        domain_idx = np.argmax(domain_proba)

        intent_conf = np.max(intent_proba)
        domain_conf = np.max(domain_proba)

        intent = intent_encoder.inverse_transform([intent_idx])[0]
        domain = domain_encoder.inverse_transform([domain_idx])[0]

        # Display results
        st.subheader("Prediction:")
        if intent_conf < 0.5 and domain_conf < 0.5:
            st.warning("⚠️ Warning: The input may be an outlier or the model is uncertain.")

        st.markdown(f"**Predicted Intent:** `{intent}` (Confidence: `{intent_conf:.2f}`)")
        st.markdown(f"**Predicted Domain:** `{domain}` (Confidence: `{domain_conf:.2f}`)")
