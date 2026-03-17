import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ==========================================
# PAGE CONFIGURATION & UI ENHANCEMENTS
# ==========================================
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")

# Custom styling for a modern dashboard look
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ff4b4b;}
    .stButton>button {width: 100%; border-radius: 5px; background-color: #ff4b4b; color: white;}
    </style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED FUNCTIONS FOR PERFORMANCE
# ==========================================
@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(r"C:\Users\Administrator\PyCharmMiscProject\WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 1. Notebook Preprocessing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df = df.dropna(subset=["TotalCharges"])

    # Convert Target to Binary
    if df["Churn"].dtype == 'O':
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID as it shouldn't be used for modeling (prevents high cardinality noise)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df


@st.cache_resource
def train_models(df):
    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-Hot Encode categorical columns (matching notebook)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train-test split (80-20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = auc(fpr, tpr)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score
        }

    return models, results, X_train.columns, scaler, numeric_cols, X


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to", ["1. Dataset Overview", "2. Model Evaluation", "3. Predict Churn"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Telco Churn CSV", type=['csv'])

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
    models, results, feature_cols, scaler, numeric_cols, X_raw = train_models(df)

    # ==========================================
    # 1. DATASET OVERVIEW
    # ==========================================
    if menu == "1. Dataset Overview":
        st.title("📊 Dataset Overview & Insights")
        st.markdown("A quick look at the cleaned and preprocessed dataset.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", df.shape[0])
        col2.metric("Total Features", df.shape[1] - 1)
        churn_rate = (df['Churn'].sum() / df.shape[0]) * 100
        col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Target Variable Distribution")
        fig_target = px.pie(df, names='Churn', title="Churn Distribution (0 = No, 1 = Yes)", hole=0.4,
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig_target, use_container_width=True)

    # ==========================================
    # 2. MODEL EVALUATION
    # ==========================================
    elif menu == "2. Model Evaluation":
        st.title("⚙️ Model Training & Evaluation")

        # Performance Comparison Chart
        st.subheader("Model Comparison (Accuracy & AUC)")
        eval_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [res["accuracy"] for res in results.values()],
            "AUC Score": [res["auc"] for res in results.values()]
        })
        fig_comp = px.bar(eval_df, x="Model", y=["Accuracy", "AUC Score"], barmode='group',
                          title="Accuracy & AUC by Model", color_discrete_sequence=['#3498db', '#9b59b6'])
        st.plotly_chart(fig_comp, use_container_width=True)

        # Detailed Analysis selection
        st.markdown("---")
        selected_model = st.selectbox("Select a model for detailed analysis:", list(results.keys()))

        col1, col2 = st.columns(2)

        with col1:
            # Confusion Matrix
            st.subheader(f"Confusion Matrix: {selected_model}")
            cm = results[selected_model]["confusion_matrix"]
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                               x=['Not Churn', 'Churn'], y=['Not Churn', 'Churn'], color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # ROC Curve
            st.subheader(f"ROC Curves")
            fig_roc = go.Figure()
            for name, res in results.items():
                fig_roc.add_trace(
                    go.Scatter(x=res["fpr"], y=res["tpr"], mode='lines', name=f"{name} (AUC={res['auc']:.2f})"))
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=True)
            st.plotly_chart(fig_roc, use_container_width=True)

        # Feature Importance for Random Forest or XGBoost
        if selected_model in ["Random Forest", "XGBoost"]:
            st.markdown("---")
            st.subheader(f"Feature Importance ({selected_model})")
            model = results[selected_model]["model"]
            importance = model.feature_importances_
            feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance}).sort_values(by="Importance",
                                                                                                    ascending=False).head(
                15)
            fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title="Top 15 Features",
                              color="Importance", color_continuous_scale='Reds')
            fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_feat, use_container_width=True)

        elif selected_model == "Logistic Regression":
            st.markdown("---")
            st.subheader("Top Logistic Regression Coefficients")
            model = results[selected_model]["model"]
            coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": model.coef_[0]}).sort_values(
                by="Coefficient", ascending=False)
            colA, colB = st.columns(2)
            with colA:
                fig_pos = px.bar(coef_df.head(10), x="Coefficient", y="Feature", orientation='h',
                                 title="Top Positive Drivers (Increases Churn)")
                fig_pos.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_pos, use_container_width=True)
            with colB:
                fig_neg = px.bar(coef_df.tail(10), x="Coefficient", y="Feature", orientation='h',
                                 title="Top Negative Drivers (Reduces Churn)")
                fig_neg.update_layout(yaxis={'categoryorder': 'total descending'})
                st.plotly_chart(fig_neg, use_container_width=True)

    # ==========================================
    # 3. PREDICT CHURN 
    # ==========================================
    elif menu == "3. Predict Churn":
        st.title("🔮 Predict Customer Churn")
        st.markdown("Enter customer details below to predict their likelihood of churning using the **XGBoost** model.")

        with st.form("prediction_form"):
            st.subheader("Customer Demographics & Account Info")
            col1, col2, col3, col4 = st.columns(4)
            gender = col1.selectbox("Gender", ["Male", "Female"])
            senior = col2.selectbox("Senior Citizen", [0, 1])
            partner = col3.selectbox("Partner", ["Yes", "No"])
            dependents = col4.selectbox("Dependents", ["Yes", "No"])

            tenure = col1.slider("Tenure (Months)", 0, 75, 12)
            monthly_charges = col2.number_input("Monthly Charges ($)", value=50.0)
            total_charges = col3.number_input("Total Charges ($)", value=500.0)
            contract = col4.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

            st.subheader("Services Subscribed")
            col5, col6, col7, col8 = st.columns(4)
            phone = col5.selectbox("Phone Service", ["Yes", "No"])
            mutiple = col6.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = col7.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = col8.selectbox("Online Security", ["Yes", "No", "No internet service"])

            device = col5.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech = col6.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            stream_tv = col7.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            stream_mov = col8.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            paperless = col5.selectbox("Paperless Billing", ["Yes", "No"])
            payment = col6.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                        "Credit card (automatic)"])

            submit = st.form_submit_button("Predict Churn Probability")

        if submit:
            # Prepare input data
            input_dict = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone, "MultipleLines": mutiple, "InternetService": internet,
                "OnlineSecurity": security, "DeviceProtection": device, "TechSupport": tech, "StreamingTV": stream_tv,
                "StreamingMovies": stream_mov, "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
            }
            input_df = pd.DataFrame([input_dict])

            # Match dummy variables
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

            # Scale numerics
            input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

            # Predict using best model (XGBoost)
            best_model = results["XGBoost"]["model"]
            prediction = best_model.predict(input_encoded)[0]
            probability = best_model.predict_proba(input_encoded)[0][1]

            st.markdown("---")
            if prediction == 1:
                st.error(f"🚨 **High Risk of Churn!** (Probability: {probability * 100:.1f}%)")
            else:
                st.success(f"✅ **Customer is likely to stay.** (Probability to churn: {probability * 100:.1f}%)")

else:
    st.info(
        "👈 Please upload the  file from the sidebar to start the application.")
    st.markdown("### Welcome to the Customer Churn Predictor!")
    st.markdown("This end-to-end Machine Learning dashboard will:")
    st.markdown("- Clean and preprocess your data automatically.")
    st.markdown("- Train 4 distinct ML algorithms (LogReg, Decision Tree, Random Forest, XGBoost).")
    st.markdown("- Provide comparative interactive visualizations.")
    st.markdown("- Allow you to test new customer profiles via an interactive prediction UI.")