import streamlit as st
import shap
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import sweetviz as sv
import streamlit.components.v1 as components  # For embedding HTML

# Load the encoded dataset for model training
customer = pd.read_csv("fyp.csv")

# Preprocessing: Drop 'Satisfaction Level' and 'Customer ID'
X = customer.drop(["Satisfaction Level", "Customer ID"], axis=1)
y = customer['Satisfaction Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Load the original dataset (for displaying user input options)
original_df = pd.read_csv("E-commerce Customer Behavior.csv")

# Generate the Sweetviz report
report = sv.analyze(customer, target_feat='Satisfaction Level', feat_cfg=config)
report.show_html("Customer_Report.html")  # Save the report

# Display the Sweetviz report in Streamlit
st.title("Customer Satisfaction Overview with Sweetviz Report")
st.write("Below is the auto-generated analysis report of the dataset.")

# Embed the Sweetviz HTML report in Streamlit
with open("Customer_Report.html", "r", encoding="utf-8") as f:
    report_html = f.read()

# Display the report in the app
components.html(report_html, height=800, scrolling=True)

# Continue with model training and SHAP explanation as before...
# Model training with LightGBM
clf = LGBMClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer using LightGBM model
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app title
st.title("Customer Satisfaction Prediction Tool")

# Part 1: Overview of Customer Satisfaction
st.header("Customer Satisfaction Overview")
st.write("This section provides a general overview of customer satisfaction predictions.")

# Display overall satisfaction insights
st.text("Satisfaction Summary")
report = classification_report(y_test, y_pred, output_dict=True)
st.write(f"**Overall Satisfaction**: {report['accuracy']:.2%}")
st.write(f"**Satisfied Customers**: {report['1']['support']} out of {len(y_test)}")
st.write(f"**Unsatisfied Customers**: {report['2']['support']} out of {len(y_test)}")
st.write(f"**Neutral Customers**: {report['0']['support']} out of {len(y_test)}")

# Summary plot (with simple explanation)
st.subheader("Visualizing Key Drivers of Satisfaction")
st.write("This plot shows the main factors influencing customer satisfaction across all predictions.")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)
fig, ax = plt.subplots()
shap.summary_plot(shap_values[0], X_test)
st.pyplot(fig)
# Part 2: Individual Customer Satisfaction Prediction
st.header("Predict Satisfaction for an Individual Customer")
st.write("Enter the customer's details to predict their satisfaction level.")

# User input based on original dataset (with categorical values)
input_data = {}

for feature in original_df.columns:
    if feature in encoders:  # If the feature was label-encoded
        if feature == 'Satisfaction Level':
           continue  # Skip 'Customer ID' and 'Satisfaction Level'
        # Let user select original categorical values (before encoding)
        unique_vals = original_df[feature].unique().tolist()
        input_data[feature] = st.selectbox(f"Select {feature}:", unique_vals)
    elif feature == 'Customer ID':
        continue  # Skip 'Customer ID'
    elif feature in ["Age", "Items Purchased", "Days Since Last Purchase"]:
        # For specific columns, allow only integer inputs
        input_data[feature] = st.number_input(f"Enter {feature}:", value=int(original_df[feature].mean()), step=1)
    else:
        # Use numeric input for non-categorical columns
        input_data[feature] = st.number_input(f"Enter {feature}:", value=original_df[feature].mean())

# Convert user input to a DataFrame
input_df = pd.DataFrame([input_data])

# Encode the categorical columns using the saved LabelEncoders
for feature, encoder in encoders.items():
    if feature in input_df.columns:
        input_df[feature] = encoder.transform(input_df[feature])

# Drop 'Customer ID' and any unnecessary columns
input_df = input_df.drop(['Customer ID','Satisfaction Level'], axis=1, errors='ignore')

# Make prediction based on the encoded user input
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]

# Display the prediction in simple terms
st.write("### Prediction Result")
if prediction == 1:
    st.write(f"**The customer is likely to be Satisfied.**")
elif prediction == 2:
    st.write(f"**The customer is likely to be Unsatisfied.**")
elif prediction == 0:
    st.write(f"**The customer is likely to be Neutral.**")

st.write(f"**Satisfaction Probability:** {probability:.2f}")

# SHAP explanation for the input (with a non-technical explanation)
st.subheader("Why this prediction?")
st.write("Below is a visual explanation of the main factors influencing the prediction for this specific customer.")
st.write("The graph helps show which details increased or decreased the likelihood of the customer being satisfied.")
shap_values_input = explainer.shap_values(input_df)

# Force plot
st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

# Decision plot
st.write("Here's another view of how the decision was made:")
st_shap(shap.decision_plot(explainer.expected_value[0], shap_values_input[0], X_test.columns))


