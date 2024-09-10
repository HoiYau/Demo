import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the dataset (assuming it's in the same directory)
customer = pd.read_csv("fyp.csv")

# Preprocessing
X = customer.drop("Satisfaction Level", axis=1)
y = customer['Satisfaction Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Satisfactory Level")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.dataframe(classification_report(y_pred, y_test,output_dict=True))

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Summary plot for class 0
st.subheader("Summary Plot for Class 0")
fig, ax = plt.subplots()
shap.summary_plot(shap_values[0], X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    if feature in ['Gender', 'Age', 'City' 'Membership Type',
                   'Total Spend', 'Items Purchased', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']:
        input_data[feature] = st.number_input(f"Enter {feature}:", value=int(X_test[feature].mean()), step=1)
    else:  # For other features, keep the original input type
        input_data[feature] = st.number_input(f"Enter {feature}:", value=X_test[feature].mean())


# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1] 

# Display prediction
if prediction == 1:
    st.write(f"**Prediction:** Satisfied")
elif prediction == 2:
    st.write(f"**Prediction:** Unsatisfied")
elif prediction == 0:
    st.write(f"**Prediction:** Neutral")

st.write(f"**Satisfaction Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)


# Force plot
st.subheader("Force Plot")
# fig, ax = plt.subplots()
# shap.plots.force(explainer.expected_value[0], shap_values_input[0,:], input_df.iloc[0,:], matplotlib=True)
st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

# st.write(input_df)
# st.pyplot(fig,bbox_inches='tight')

# Decision plot
st.subheader("Decision Plot")
# fig, ax = plt.subplots()
# shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns)
st_shap(shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns))
# st.pyplot(fig)
