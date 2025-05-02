# HouseWorthAI - Streamlit app with complete ML pipeline and dark retro UI
# All-in-one file with CSV upload and online data fetching support

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import requests
import io

# Page configuration
st.set_page_config(page_title="HouseWorthAI", layout="wide", initial_sidebar_state="expanded")

# Inject CSS for neon pixel-art theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
* { font-family: 'VT323', monospace; }
.main { background: #000000; color: white; overflow-y: auto; }
h1, h2, h3, h4 { color: #ff00ff; text-shadow: 0 0 10px #ff00ff; font-family: 'Press Start 2P'; }
.stButton>button, .stDownloadButton>button {
    background: black; color: #00ffff; border: 2px solid #00ffff;
    font-family: 'Press Start 2P'; text-transform: uppercase;
    transition: 0.3s ease; animation: pulse 2s infinite;
}
.stButton>button:hover { background: #00ffff; color: black; }
@keyframes pulse {
  0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="HouseWorthAI 🏠",
        options=["Home", "Upload/Fetch Data", "Preprocessing", "Modeling", "Evaluation", "Download"],
        icons=["house", "cloud-upload", "gear", "robot", "bar-chart", "download"],
        default_index=0
    )

# Session state for persistent data
for key in ["df", "X_train", "X_test", "y_train", "y_test", "model", "y_pred"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Home page
if selected == "Home":
    st.title("🏠 HouseWorthAI")
    st.markdown("""
    <h2 style='color:#00ffff;'>Real Estate Price Prediction Engine</h2>
    <p>Upload or fetch housing datasets, train models, and predict with style ✨</p>
    """, unsafe_allow_html=True)

# Upload or fetch dataset
elif selected == "Upload/Fetch Data":
    st.header("📊 Upload or Fetch Dataset")

    option = st.radio("Select Data Source:", ["Upload CSV", "Fetch Sample Data Online"])

    if option == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            st.session_state.df = pd.read_csv(file)
            st.success("Data uploaded successfully!")

    else:
        st.info("Fetching sample housing dataset from online GitHub repo...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        response = requests.get(url).content
        st.session_state.df = pd.read_csv(io.StringIO(response.decode("utf-8")))
        st.success("Sample data loaded!")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head())

# Preprocessing
elif selected == "Preprocessing":
    st.header("🔧 Data Preprocessing")
    df = st.session_state.df
    if df is None:
        st.warning("Upload or fetch data first.")
    else:
        if df.isnull().sum().sum() > 0:
            st.write("Missing values detected. Filling with mean...")
            df.fillna(df.mean(numeric_only=True), inplace=True)

        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols):
            st.write(f"Encoding categorical columns: {list(cat_cols)}")
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])

        st.dataframe(df.head())
        st.session_state.df = df

# Modeling
elif selected == "Modeling":
    st.header("🤖 Model Training")
    df = st.session_state.df
    if df is None:
        st.warning("Please preprocess data first.")
    else:
        target = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
        features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

        if features:
            X = df[features]
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state.update({
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
                "model": model, "y_pred": y_pred
            })

            st.success(f"{model_choice} model trained successfully!")

# Evaluation
elif selected == "Evaluation":
    st.header("📈 Model Evaluation")
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    if y_test is None or y_pred is None:
        st.warning("Train a model first.")
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        st.markdown(f"**R² Score:** `{r2:.4f}`")
        st.markdown(f"**RMSE:** `{rmse:.4f}`")
        st.markdown(f"**MAE:** `{mae:.4f}`")

        result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig = px.scatter(result_df, x="Actual", y="Predicted", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

# Download Results
elif selected == "Download":
    st.header("📤 Download Predictions")
    if st.session_state.y_pred is not None:
        result_df = pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": st.session_state.y_pred})
        st.download_button("Download CSV", result_df.to_csv(index=False), "houseworth_predictions.csv", "text/csv")
    else:
        st.warning("No predictions to download yet.")
