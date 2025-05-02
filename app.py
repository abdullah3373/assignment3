# EduInsights - Student Performance Prediction Streamlit App
# All-in-one file with CSV upload, preprocessing, ML modeling, evaluation, and enhanced visuals

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests

# Page setup
st.set_page_config(page_title="EduInsights", layout="wide", initial_sidebar_state="expanded")

# Neon theme styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
* { font-family: 'VT323', monospace; }
.main { background: #000000; color: white; }
h1, h2, h3, h4 { color: #00ffff; text-shadow: 0 0 10px #00ffff; font-family: 'Press Start 2P'; }
.stButton>button, .stDownloadButton>button {
    background: black; color: #00ff00; border: 2px solid #00ff00;
    font-family: 'Press Start 2P'; text-transform: uppercase;
    transition: 0.3s ease; animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="EduInsights 📚",
        options=["Home", "Upload/Fetch Data", "Visualize", "Preprocessing", "Modeling", "Evaluation", "Download"],
        icons=["house", "cloud-upload", "bar-chart", "gear", "robot", "activity", "download"],
        default_index=0
    )

# Session storage
for key in ["df", "X_train", "X_test", "y_train", "y_test", "model", "y_pred"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Home page
if selected == "Home":
    st.title("📚 EduInsights")
    st.markdown("""
    <h2 style='color:#ff00ff;'>Predict Student Academic Performance</h2>
    <p>Analyze student data, visualize insights, and build predictive models – all in one neon-themed app! 🌟</p>
    """, unsafe_allow_html=True)

# Upload or Fetch data
elif selected == "Upload/Fetch Data":
    st.header("📂 Upload or Fetch Student Data")
    option = st.radio("Choose Data Source:", ["Upload CSV", "Fetch Demo Dataset"])

    if option == "Upload CSV":
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file:
            st.session_state.df = pd.read_csv(file)
            st.success("Data uploaded successfully!")
    else:
        st.info("Fetching student performance dataset from UCI repo...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/StudentPerformance.csv"
        content = requests.get(url).content
        st.session_state.df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        st.success("Demo data loaded!")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())

# Visualize data
elif selected == "Visualize":
    st.header("📊 Data Visualizations")
    df = st.session_state.df
    if df is None:
        st.warning("Please upload or fetch a dataset first.")
    else:
        st.subheader("Grade Distribution")
        if 'G3' in df.columns:
            fig1 = px.histogram(df, x='G3', nbins=20, title='Final Grade Distribution', template='plotly_dark')
            st.plotly_chart(fig1)

        st.subheader("Gender vs Performance")
        if 'sex' in df.columns and 'G3' in df.columns:
            fig2 = px.box(df, x='sex', y='G3', color='sex', title='Gender vs Final Grade', template='plotly_dark')
            st.plotly_chart(fig2)

        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=np.number)
        if len(num_df.columns) > 1:
            fig3 = px.imshow(num_df.corr(), text_auto=True, title='Numeric Feature Correlation', template='plotly_dark')
            st.plotly_chart(fig3)

# Preprocessing
elif selected == "Preprocessing":
    st.header("⚙️ Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("Upload or fetch data first.")
    else:
        if df.isnull().sum().sum() > 0:
            st.write("Filling missing values with mean...")
            df.fillna(df.mean(numeric_only=True), inplace=True)

        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

        st.session_state.df = df
        st.success("Preprocessing complete.")
        st.dataframe(df.head())

# Modeling
elif selected == "Modeling":
    st.header("🤖 Model Training")
    df = st.session_state.df
    if df is None:
        st.warning("Preprocess data first.")
    else:
        target = st.selectbox("Select Target Column", df.columns, index=-1)
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

        if features:
            X = df[features]
            y = df[target]
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.update({
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "model": model, "y_pred": y_pred
            })
            st.success(f"{model_type} model trained.")

# Evaluation
elif selected == "Evaluation":
    st.header("📈 Model Evaluation")
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    if y_test is None or y_pred is None:
        st.warning("Please train a model first.")
    else:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.metric("Accuracy", f"{acc:.2f}")
        st.metric("Precision", f"{prec:.2f}")
        st.metric("Recall", f"{rec:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")

        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig = px.imshow(cm, text_auto=True, title='Confusion Matrix', labels=dict(x="Predicted", y="Actual"), template='plotly_dark')
        st.plotly_chart(fig)

# Download
elif selected == "Download":
    st.header("⬇️ Download Results")
    if st.session_state.y_pred is not None:
        df_out = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": st.session_state.y_pred})
        st.download_button("Download Predictions", df_out.to_csv(index=False), "eduinsights_predictions.csv", "text/csv")
    else:
        st.warning("No predictions to download yet.")
