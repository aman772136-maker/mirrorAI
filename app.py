import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Mirror AI | Bias Bounty", page_icon="🪞")

@st.cache_resource
def load_model():
    return joblib.load('mirror_model.joblib')

model = load_model()

st.title("🪞 Mirror AI: Bias Bounty")
st.markdown("Test the model with different identity terms to uncover algorithmic bias.")

user_input = st.text_input("Enter a statement to evaluate:", value="I am a woman")

if st.button("Analyze Text", type="primary"):
    pred = model.predict([user_input])[0]
    prob = model.predict_proba([user_input])[0].max()
    
    if pred == 1:
        st.error(f"🚨 **Flagged as Toxic** (Confidence: {prob:.2%})")
        st.warning("Bias detected: A harmless identity term was classified as toxic!")
    else:
        st.success(f"✅ **Safe** (Confidence: {prob:.2%})")

with st.expander("🔍 Inspect Training Data"):
    df = pd.read_csv('bias_bounty_dataset.csv')
    st.dataframe(df.head(20))
