#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# Updated data for confusion matrices
confusion_matrices = {
    'OLS': [
        [-1.0, 0.0, 1.0, 2.0, 3.0, 197142903.0],
        [-1.0, 0, 0, 0, 0, 0],
        [0.0, 5, 1137, 170, 1, 0],
        [1.0, 1, 10, 36, 2, 1],
        [2.0, 0, 0, 0, 0, 0],
        [3.0, 0, 0, 0, 0, 0],
        [197142903.0, 0, 0, 0, 0, 0]
    ],
    'Ridge': [
        [0, 0, 0, 0, 0, 0],
        [5, 1129, 179, 0, 0, 0],
        [0, 7, 41, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    'Lasso': [
        [826, 487, 0],
        [20, 30, 1],
        [0, 0, 0]
    ],
    'RFE Logistic Regression': [
        [726, 4],
        [587, 47]
    ],
    'Polynomial Logistic Regression': [
        [1106, 207],
        [30, 21]
    ]
}

# Example data for bar graphs
metrics = {
    'Model': ['Ridge', 'Lasso', 'Logistic Regression', 'Polynomial Logistic', 'OLS'],
    'Accuracy': [0.85, 0.80, 0.82, 0.88, 0.86],
    'Recall': [0.80, 0.75, 0.78, 0.85, 0.87],
    'Precision': [0.83, 0.78, 0.80, 0.87, 0.99],
    'F1 Score': [0.82, 0.76, 0.79, 0.86, 0.92]
}
df_metrics = pd.DataFrame(metrics)

def plot_confusion_matrix(cm, model_name, normalize=False, log_scale=False, cap_value=None):
    try:
        # Convert to numpy array for easier manipulation
        cm = np.array(cm)

        # Apply capping if cap_value is provided
        if cap_value is not None:
            cm[cm > cap_value] = cap_value

        # Apply normalization if normalize is set to True
        if normalize:
            max_val = np.max(cm)
            if max_val > 0:
                cm = cm / max_val  # Normalize by dividing by the maximum value

        # Apply log scale if log_scale is set to True
        if log_scale:
            cm = np.log1p(cm)  # log1p to avoid log(0) issues

        # Create heatmap
        z = cm
        x = [str(i) for i in range(len(cm[0]))]
        y = [str(i) for i in range(len(cm))]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
        fig.update_layout(title=f'Confusion Matrix for {model_name}')

        # Plot the figure in Streamlit
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error plotting confusion matrix for {model_name}: {e}")

def plot_bar_graph(metric):
    try:
        fig = px.bar(df_metrics, x='Model', y=metric, color='Model', title=f'{metric} Comparison', range_y=[0.5, 1.0])
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting bar graph for {metric}: {e}")

def plot_histogram(data, title):
    try:
        fig = px.histogram(data, nbins=30, title=title)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error plotting histogram for {title}: {e}")

st.title('Bankruptcy Prediction Model Comparison')

# Display confusion matrices with additional features (normalize, log scale, or cap outliers)
for model_name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, model_name, normalize=True, log_scale=False, cap_value=1e6)

# Display bar graphs for metrics
metrics_list = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
for metric in metrics_list:
    plot_bar_graph(metric)

# Example data for histograms
hist_data = {
    'Model Accuracy Comparison': [0.85, 0.80, 0.82, 0.88, 0.86],
    'Model Recall Comparison': [0.80, 0.75, 0.78, 0.85, 0.87],
    'Model Precision Comparison': [0.83, 0.78, 0.80, 0.87, 0.99],
    'Model F1 Score Comparison': [0.82, 0.76, 0.79, 0.86, 0.92]
}

# Display histograms
for title, data in hist_data.items():
    plot_histogram(data, title)


# In[ ]:




