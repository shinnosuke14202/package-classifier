import streamlit as st
import joblib

# Load your trained model
model = joblib.load("./model/app_game_classifier_model.pkl")


def predict_category(package_name):

    # Make prediction
    prediction = "Game" if model.predict([package_name])[0] == 1 else "No Game"

    # Get confidence score
    probabilities = model.predict_proba([package_name])[0]
    confidence = max(probabilities)

    return prediction, confidence

def main():
    st.header("📦 Batch Package Category Checker")

    package_text = st.text_area("Enter one package name per line", height=300)

    if st.button("Check"):
        packages = [p.strip() for p in package_text.split('\n') if p.strip()]
        
        if not packages:
            st.warning("Please enter at least one package name.")
        else:
            st.subheader("Results:")
            for pkg in packages:
                category, confidence = predict_category(pkg)
                st.markdown(f"• `{pkg}` → **{category}** ({confidence:.2%} confidence)")

if __name__ == '__main__':
    main()


