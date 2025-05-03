"""
AF3005 Assignment 3: Financial ML Dashboard
Instructor: Dr. Usama Arshad
BS Financial Technology - Spring 2025

How to Run:
1. pip install -r requirements.txt
2. streamlit run finml_app.py
3. Use sidebar to load data (Kragle CSV/Yahoo Finance)
4. Follow step-by-step workflow using buttons
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    confusion_matrix, silhouette_score
)
import base64

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="FinML Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM THEME
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;500&display=swap');
* {font-family: 'Roboto Mono', monospace;}
.stApp {background: #0F0F23; color: #00FF9D;}
h1, h2, h3 {color: #00D1FF; border-bottom: 2px solid #FF00E5;}
.stButton>button {
    background: #1A1A2F !important;
    color: #00FF9D !important;
    border: 2px solid #00FF9D !important;
    border-radius: 5px;
    transition: 0.3s;
}
.stButton>button:hover {transform: scale(1.05);}
.success {color: #00FF00 !important;}
.warning {color: #FFA500 !important;}
</style>
""", unsafe_allow_html=True)

# ======================
# SESSION STATE
# ======================
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# ======================
# HELPER FUNCTIONS
# ======================
def add_gif(url, width=300):
    st.markdown(f'<img src="{url}" width="{width}">', unsafe_allow_html=True)

def progress_step():
    st.session_state.current_step += 1

# ======================
# SIDEBAR - DATA LOADING
# ======================
with st.sidebar:
    st.header("📈 Data Configuration")
    data_source = st.radio("Select Data Source:", 
                          ["Upload Kragle Dataset", "Yahoo Finance"])
    
    if data_source == "Yahoo Finance":
        ticker = st.text_input("Stock Ticker (e.g., AAPL):", "AAPL")
        start_date = st.date_input("Start Date:", datetime(2020, 1, 1))
        end_date = st.date_input("End Date:", datetime.today())
        
        if st.button("Fetch Market Data"):
            with st.spinner("Downloading financial data..."):
                try:
                    df = yf.download(ticker, start=start_date, end=end_date)
                    df = df.reset_index()
                    df['Daily Return'] = df['Close'].pct_change()
                    st.session_state.df = df.dropna()
                    st.success("Yahoo Finance data loaded!")
                    st.session_state.current_step = 2
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
    
    elif data_source == "Upload Kragle Dataset":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file and st.button("Process Kragle Data"):
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success("Kragle dataset loaded!")
                st.session_state.current_step = 2
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

# ======================
# MAIN WORKFLOW
# ======================
st.title("💻 Financial Machine Learning Workflow")
add_gif("https://i.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.webp", 400)

# ======================
# STEP 1: DATA OVERVIEW
# ======================
if st.session_state.current_step >= 1:
    st.header("1. Data Overview")
    if st.session_state.df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state.df.head())
        
        st.subheader("Price Movement")
        fig = px.line(st.session_state.df, x='Date', y='Close', 
                     title='Stock Price Movement', 
                     template='plotly_dark',
                     color_discrete_sequence=['#00FF9D'])
        st.plotly_chart(fig)
    else:
        st.warning("Please load data first using the sidebar controls")

# ======================
# STEP 2: PREPROCESSING
# ======================
if st.session_state.current_step >= 2:
    st.header("2. Data Preprocessing")
    if st.button("Run Data Preprocessing"):
        df = st.session_state.df.copy()
        
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            initial_rows = len(df)
            df = df.dropna()
            st.success(f"Removed {initial_rows - len(df)} rows with missing values")
        
        # Feature engineering
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['Daily Return'].rolling(30).std()
        
        st.session_state.df = df.dropna()
        progress_step()
        st.success("Preprocessing complete! Added technical indicators")
        
        st.subheader("Processed Data")
        st.dataframe(st.session_state.df.tail())

# ======================
# STEP 3: FEATURE ENGINEERING
# ======================
if st.session_state.current_step >= 3:
    st.header("3. Feature Engineering")
    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        selected_features = st.multiselect("Select Features for Modeling",
                                         numeric_cols,
                                         default=numeric_cols[:-1])
        target_var = st.selectbox("Select Target Variable", numeric_cols)
        
        if st.button("Confirm Features"):
            st.session_state.X = df[selected_features]
            st.session_state.y = df[target_var]
            progress_step()
            st.success("Features configured!")

# ======================
# STEP 4: MODEL TRAINING
# ======================
if st.session_state.current_step >= 4:
    st.header("4. Model Configuration")
    model_type = st.selectbox("Select ML Model",
                             ["Linear Regression", "Logistic Regression", "K-Means Clustering"])
    
    # Model parameters
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        model = KMeans(n_clusters=n_clusters)
    
    test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
    
    if st.button("Train Model"):
        X = StandardScaler().fit_transform(st.session_state.X)
        y = st.session_state.y
        
        if model_type != "K-Means Clustering":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Show train-test split
            st.subheader("Data Split Ratio")
            split_df = pd.DataFrame({
                'Set': ['Training', 'Testing'],
                'Samples': [len(y_train), len(y_test)]
            })
            fig_split = px.pie(split_df, names='Set', values='Samples', 
                              title='Train/Test Data Distribution',
                              template='plotly_dark',
                              color_discrete_sequence=px.colors.qualitative.Dark2)
            st.plotly_chart(fig_split)
        else:
            model.fit(X)
            y_pred = model.predict(X)
        
        st.session_state.model = model
        st.session_state.y_pred = y_pred
        if model_type != "K-Means Clustering":
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        
        progress_step()
        st.success(f"{model_type} training complete!")

# ======================
# STEP 5: EVALUATION
# ======================
if st.session_state.current_step >= 5:
    st.header("5. Model Evaluation")
    if st.session_state.model is not None:
        model = st.session_state.model
        model_type = type(model).__name__
        
        if model_type == "LinearRegression":
            st.subheader("Regression Metrics")
            rmse = np.sqrt(mean_squared_error(st.session_state.y_test, st.session_state.y_pred))
            r2 = r2_score(st.session_state.y_test, st.session_state.y_pred)
            
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")
            
            fig = px.scatter(x=st.session_state.y_test, y=st.session_state.y_pred,
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Actual vs Predicted Values",
                            template='plotly_dark',
                            color_discrete_sequence=['#FF00E5'])
            st.plotly_chart(fig)
        
        elif model_type == "LogisticRegression":
            st.subheader("Classification Metrics")
            acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            
            st.metric("Accuracy", f"{acc:.2%}")
            fig = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix",
                           template='plotly_dark')
            st.plotly_chart(fig)
        
        elif model_type == "KMeans":
            st.subheader("Clustering Results")
            silhouette = silhouette_score(st.session_state.X, st.session_state.y_pred)
            st.metric("Silhouette Score", f"{silhouette:.2f}")
            
            fig = px.scatter_3d(st.session_state.X,
                               x=st.session_state.X[:,0],
                               y=st.session_state.X[:,1],
                               z=st.session_state.X[:,2],
                               color=st.session_state.y_pred,
                               title="Cluster Visualization",
                               template='plotly_dark')
            st.plotly_chart(fig)

# ======================
# BONUS FEATURES
# ======================
st.sidebar.markdown("---")
st.sidebar.header("Bonus Features")
if st.session_state.df is not None:
    st.sidebar.download_button("Download Processed Data",
                              st.session_state.df.to_csv(index=False),
                              "processed_financial_data.csv",
                              "text/csv")

if st.session_state.model is not None:
    st.sidebar.markdown("### Model Coefficients")
    try:
        if hasattr(st.session_state.model, 'coef_'):
            coefs = pd.DataFrame({
                'Feature': st.session_state.X.columns,
                'Importance': st.session_state.model.coef_[0]
            })
            st.sidebar.dataframe(coefs.sort_values('Importance', ascending=False))
    except Exception as e:
        st.sidebar.warning("Coefficients not available for this model")

# ======================
# DOCUMENTATION
# ======================
st.sidebar.markdown("---")
st.sidebar.info("""
**How to Use:**
1. Select data source in sidebar
2. Complete steps sequentially
3. Each step unlocks next
4. Visualizations update automatically
""")
