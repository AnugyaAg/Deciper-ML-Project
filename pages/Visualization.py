import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import load_model, get_model_info
import os
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
import joblib

def show():
    st.title("Model Visualization & Monitoring")

    # Set up directories
    models_dir = Path("models")
    datasets_dir = Path("datasets")

    # Check if models and datasets exist
    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        st.warning("No trained models found. Please train a model first.")
        return

    if not datasets_dir.exists() or not any(datasets_dir.glob("*.csv")):
        st.warning("No datasets available. Please upload a dataset first.")
        return

    # Sidebar for selecting model and dataset
    with st.sidebar:
        st.subheader("Select Model & Dataset")
        model_files = list(models_dir.glob("*.pkl"))
        dataset_files = list(datasets_dir.glob("*.csv"))

        selected_model = st.selectbox(
            "Select Model",
            options=[f.name for f in model_files],
            format_func=lambda x: x.replace(".pkl", "").replace("_", " ").title()
        )

        selected_dataset = st.selectbox(
            "Select Dataset",
            options=[f.name for f in dataset_files],
            format_func=lambda x: x.replace(".csv", "").title()
        )

    # Proceed if both selections are made
    if selected_model and selected_dataset:
        try:
            # Load model and dataset
            model_data = joblib.load(models_dir / selected_model)
            model = model_data['model']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']
            df = pd.read_csv(datasets_dir / selected_dataset)

            # Display model info
            model_info = get_model_info(models_dir / selected_model)

            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Model Type:", model_info["type"])
                st.write("Parameters:", model_info["parameters"])
            with col2:
                st.write("Dataset Shape:", df.shape)
                st.write("Features:", len(df.columns))

            # Feature Importance (if available)
            if model_info["feature_importance"] is not None:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_info["feature_importance"]
                }).sort_values('Importance', ascending=False)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig)

            # SHAP Values for Feature Contribution Analysis
            st.subheader("SHAP Values (Feature Contributions)")
            try:
                X = df[feature_names].values
                X_sample = X[:50]  # use first 50 rows for explanation
                background = shap.kmeans(X, 10)  # background for kernel explainer

                model_type = type(model).__name__.lower()

                # Use different explainers based on model type
                if "randomforest" in model_type or "xgb" in model_type:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                elif "logisticregression" in model_type:
                    explainer = shap.LinearExplainer(model, X)
                    shap_values = explainer.shap_values(X_sample)

                elif hasattr(model, "predict_proba"):
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_sample)

                else:
                    st.warning("Model type not supported for SHAP analysis.")
                    shap_values = None

                if shap_values is not None:
                    st.write("SHAP Summary Plot:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                    st.pyplot(fig)

                    st.write("SHAP Bar Plot:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
                    st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not compute SHAP values: {str(e)}")

            # Data Quality Checks
            st.subheader("Data Quality Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Missing Values:")
                missing_values = df.isnull().sum()
                fig = px.bar(x=missing_values.index, y=missing_values.values)
                st.plotly_chart(fig)
            with col2:
                st.write("Data Types:")
                st.write(df.dtypes)

            # Feature Distributions
            st.subheader("Feature Distributions")
            for column in feature_names:
                fig = px.histogram(df, x=column, color=target_column, marginal="box")
                st.plotly_chart(fig)

            # Correlation Heatmap
            st.subheader("Feature Correlation Matrix")
            corr_matrix = df[feature_names + [target_column]].corr()
            fig = px.imshow(corr_matrix, color_continuous_scale="RdBu")
            st.plotly_chart(fig)

            # Model Performance
            st.subheader("Model Performance Metrics")
            X_df = df[feature_names]
            y_true = df[target_column]
            y_pred = model.predict(X_df)

            # Classification Report
            st.write("Classification Report:")
            report = classification_report(y_true, y_pred)
            st.text(report)

            # Confusion Matrix
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

            # ROC Curve
            try:
                y_pred_proba = model.predict_proba(X_df)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                st.plotly_chart(fig)
            except Exception as e:
                st.warning(f"Could not generate ROC curve: {str(e)}")

            # Statistical Tests
            st.subheader("Statistical Analysis")
            for column in feature_names:
                if df[column].dtype in ['int64', 'float64']:
                    group_0 = df[df[target_column] == 0][column]
                    group_1 = df[df[target_column] == 1][column]
                    t_stat, p_value = stats.ttest_ind(group_0, group_1)
                    st.write(f"{column}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

        except Exception as e:
            st.error(f"Error processing model and data: {str(e)}")
            st.error("Please ensure the model and dataset are compatible.")
            
