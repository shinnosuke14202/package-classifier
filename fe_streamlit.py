import streamlit as st
import joblib

# 👇 Model options: {Label shown in dropdown : file path}
MODELS = {
    "Logistic Regression (v1)": "./model/logistic_regression_model_v0.pkl",
    "Logistic Regression (v2)": "./model/logistic_regression_model_v1.pkl",
    "Random Forest (v1)": "./model/random_forest_model_v0.pkl",
    "Random Forest (v2)": "./model/random_forest_model_v1.pkl"
}

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

def predict_category(package_name, model):
    prediction = "Game" if model.predict([package_name])[0] == 1 else "No Game"
    probabilities = model.predict_proba([package_name])[0]
    confidence = max(probabilities)
    return prediction, confidence

def main():
    st.set_page_config(page_title="Package Classifier")
    st.title("📦 Batch Package Category Checker")

    selected_model_name = st.selectbox("Select model", list(MODELS.keys()))
    model_path = MODELS[selected_model_name]
    model = load_model(model_path)

    package_text = st.text_area("Enter one package name per line", height=300)

    if st.button("Check"):
        packages = [p.strip() for p in package_text.split('\n') if p.strip()]
        
        if not packages:
            st.warning("Please enter at least one package name.")
        else:
            st.subheader(f"Results using: **{selected_model_name}**")
            for pkg in packages:
                category, confidence = predict_category(pkg, model)
                st.markdown(f"• `{pkg}` → **{category}** ({confidence:.2%} confidence)")

if __name__ == '__main__':
    main()
