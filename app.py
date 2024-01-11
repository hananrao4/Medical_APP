import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

cholesterol_df = pd.read_csv('cholesterol_data.csv')
lft_df = pd.read_csv('lft_data.csv')


def preprocess_data(df, target_column):
    df = pd.get_dummies(df, columns=['Gender', 'Diabetes History', 'Blood Pressure History'], drop_first=True)
    
    features = df.drop([target_column], axis=1)
    target = df[target_column]

    return df.columns, features, target

# cholesterol_Test
cholesterol_columns, cholesterol_features, cholesterol_target = preprocess_data(cholesterol_df, 'Status')
cholesterol_model = RandomForestClassifier()
cholesterol_model.fit(cholesterol_features, cholesterol_target)

#Liver Test
lft_columns, lft_features, lft_target = preprocess_data(lft_df, 'Status')
lft_model = RandomForestClassifier()
lft_model.fit(lft_features, lft_target)


st.title("Medical Test Prediction App")


test_type = st.radio("Select Test Type", ["Cholesterol Test", "Liver Function Test"])

if test_type == "Cholesterol Test":
    st.write("Cholesterol Test Data:")
    st.dataframe(cholesterol_df)

elif test_type == "Liver Function Test":
    st.write("Liver Function Test Data:")
    st.dataframe(lft_df)

st.header("Enter Test Parameters for Prediction")

user_input = {}
if test_type == "Cholesterol Test":
    user_input['Age'] = st.slider("Age", min_value=30, max_value=70, value=50)
    user_input['Gender'] = st.radio("Gender", ['Male', 'Female'])
    user_input['Diabetes History'] = st.radio("Diabetes History", ['Yes', 'No'])
    user_input['Blood Pressure History'] = st.selectbox("Blood Pressure History", ['Normal', 'High', 'Elevated'])
    user_input['Total Cholesterol'] = st.slider("Total Cholesterol", min_value=150, max_value=250, value=200)
    user_input['HDL Cholesterol'] = st.slider("HDL Cholesterol", min_value=40, max_value=70, value=55)
    user_input['LDL Cholesterol'] = st.slider("LDL Cholesterol", min_value=90, max_value=180, value=130)
    user_input['Triglycerides'] = st.slider("Triglycerides", min_value=110, max_value=220, value=150)

elif test_type == "Liver Function Test":
    user_input['Age'] = st.slider("Age", min_value=30, max_value=70, value=50)
    user_input['Gender'] = st.radio("Gender", ['Male', 'Female'])
    user_input['Blood Pressure History'] = st.selectbox("Blood Pressure History", ['Normal', 'High', 'Elevated'])
    user_input['Diabetes History'] = st.radio("Diabetes History", ['Yes', 'No'])
    user_input['Alkaline Phosphatase (ALP)'] = st.slider("ALP", min_value=50, max_value=150, value=100)
    user_input['Aspartate Aminotransferase (AST)'] = st.slider("AST", min_value=10, max_value=50, value=30)
    user_input['Alanine Aminotransferase (ALT)'] = st.slider("ALT", min_value=10, max_value=40, value=20)
    user_input['Total Bilirubin'] = st.slider("Total Bilirubin", min_value=0.1, max_value=1.2, value=0.6)
    user_input['Direct Bilirubin'] = st.slider("Direct Bilirubin", min_value=0.05, max_value=0.5, value=0.2)
    user_input['Albumin'] = st.slider("Albumin", min_value=3.5, max_value=5.5, value=4.5)

# Prediction
if st.button("Predict"):
    user_input_df = pd.DataFrame([user_input])

    if test_type == "Cholesterol Test":
        user_input_df = pd.get_dummies(user_input_df, columns=['Gender', 'Diabetes History', 'Blood Pressure History'], drop_first=True)

        user_input_df = user_input_df.reindex(columns=cholesterol_columns, fill_value=0)

        user_input_df = user_input_df.drop('Status', axis=1, errors='ignore')

        prediction = cholesterol_model.predict(user_input_df)[0]
        st.write("Cholesterol Test Prediction:", prediction)

    elif test_type == "Liver Function Test":
        user_input_df = pd.get_dummies(user_input_df, columns=['Gender', 'Diabetes History', 'Blood Pressure History'], drop_first=True)

        user_input_df = user_input_df.reindex(columns=lft_columns, fill_value=0)

        user_input_df = user_input_df.drop('Status', axis=1, errors='ignore')

        prediction = lft_model.predict(user_input_df)[0]
        st.write("Liver Function Test Prediction:", prediction)