import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("This app uses a **Logistic Regression model with TF-IDF** to classify news as **Fake** or **Real**.")

# User input
user_input = st.text_area("✍️ Paste a news article or headline below:")

if st.button("🔍 Predict"):
    if user_input.strip() != "":
        # Transform input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        proba = model.predict_proba(transformed_input)[0]

        # Display results
        if prediction == 1:  # assuming 1 = Real, 0 = Fake
            st.success(f"✅ This looks like a REAL news article! (Confidence: {proba[1]:.2f})")
        else:
            st.error(f"🚨 This looks like FAKE news! (Confidence: {proba[0]:.2f})")

        # Show probability distribution
        st.write("### Prediction Probabilities")
        st.write(f"🟢 Real: {proba[1]:.2f}")
        st.write(f"🔴 Fake: {proba[0]:.2f}")

    else:
        st.warning("⚠️ Please enter some text to analyze.")
