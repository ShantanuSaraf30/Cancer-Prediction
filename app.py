import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import numpy as np
# Load datasets
stage_data = pd.read_csv('stagepred.csv')
survival_data = pd.read_csv('survivalrate.csv')
treatment_response = pd.read_csv('treatmentresponse.csv')
chemo_data = pd.read_csv('alldata.csv')

# Function to preprocess data
def preprocess_data(data, selected_features, target_variable):
    data = data.dropna(subset=selected_features + [target_variable])
    X = data[selected_features]
    y = data[target_variable]
    X = pd.get_dummies(X)
    return X, y

# Function to train a Decision Tree Classifier
@st.cache(allow_output_mutation=True)
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to train a Logistic Regression model
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Function to train a Random Forest Classifier
@st.cache(allow_output_mutation=True)
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, None

# Function to train a Decision Tree Regressor
def train_decision_tree_regressor(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
# Sidebar navigation
page = st.sidebar.selectbox("Navigate to", ["Home", "Stage Prediction", "Survival Prediction", "Treatment Response", "Chemotherapy Prediction"])

if page == "Home":
    st.subheader('Welcome to Cancer Prediction System')
    st.write('This system helps predict cancer stages and survival status based on patient data.')

elif page == "Stage Prediction":
    st.title('Cancer Stage Prediction')

    selected_features_stage = [
        "Age recode with <1 year olds and 90+",
        "Sex",
        "Year of diagnosis",
        "Race recode (White, Black, Other)",
        "TNM 7/CS v0204+ Schema (thru 2017)",
        "Diagnostic Confirmation",
        "CS tumor size (2004-2015)",
        "CS extension (2004-2015)",
        "CS lymph nodes (2004-2015)",
        "CS mets at dx (2004-2015)"
    ]
    target_variable_stage = 'Combined Summary Stage (2004+)'

    X_stage, y_stage = preprocess_data(stage_data, selected_features_stage, target_variable_stage)
    X_train_stage, X_test_stage, y_train_stage, y_test_stage = train_test_split(X_stage, y_stage, test_size=0.2, random_state=42)

    model_stage, _ = train_random_forest(X_train_stage, y_train_stage)

    # User input form for stage prediction
    st.subheader('Enter Patient Data for Stage Prediction')
    input_data_stage = {}
    for feature in selected_features_stage:
        input_data_stage[feature] = st.selectbox(f'Select {feature}', options=[''] + list(stage_data[feature].unique()))

    if st.button('Predict Stage'):
        if "" in input_data_stage.values():
            st.warning("Please fill in all fields.")
        else:
            try:
                input_df_stage = pd.DataFrame([input_data_stage], columns=selected_features_stage)
                input_df_stage = pd.get_dummies(input_df_stage)
                missing_cols_stage = set(X_train_stage.columns) - set(input_df_stage.columns)
                for col in missing_cols_stage:
                    input_df_stage[col] = 0
                input_df_stage = input_df_stage[X_train_stage.columns]
                prediction_stage = model_stage.predict(input_df_stage)
                st.success(f'Predicted stage: {prediction_stage[0]}')
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


elif page == "Survival Prediction":
    st.title('Cancer Survival Prediction')
    
    # Load the dataset
    data = pd.read_csv('survivalrate.csv')

    # Select the specified columns and the target column
    selected_attributes_survival = [
        "Age recode with <1 year olds",
        "Sex",
        "TNM 7/CS v0204+ Schema (thru 2017)",
        "Diagnostic Confirmation",
        "Radiation recode",
        "Chemotherapy recode (yes, no/unk)",
        "Year of diagnosis",
        "Primary Site - labeled",
        "Histologic Type ICD-O-3",
        "CS tumor size (2004-2015)",
        "Survival months"
    ]

    # Preprocess the data
    data = data[data["Survival months"] != "Unknown"]
    X_survival, y_survival = preprocess_data(data, selected_attributes_survival, "Survival months")
    X_train_survival, X_test_survival, y_train_survival, y_test_survival = train_test_split(X_survival, y_survival, test_size=0.2, random_state=42)

    model_survival = train_decision_tree_regressor(X_train_survival, y_train_survival)

    # User input form for survival prediction
    st.subheader('Enter Patient Data for Survival Prediction')
    input_data_survival = {}
    for feature in selected_attributes_survival[:-1]:  # Exclude the target variable
        input_data_survival[feature] = st.selectbox(f'Select {feature}', options=[''] + list(data[feature].unique()))

    if st.button('Predict Survival Months'):
        if "" in input_data_survival.values():
            st.warning("Please fill in all fields.")
        else:
            try:
                input_df_survival = pd.DataFrame([input_data_survival], columns=selected_attributes_survival[:-1])
                input_df_survival = pd.get_dummies(input_df_survival)
                missing_cols_survival = set(X_train_survival.columns) - set(input_df_survival.columns)
                for col in missing_cols_survival:
                    input_df_survival[col] = 0
                input_df_survival = input_df_survival[X_train_survival.columns]
                prediction_survival = model_survival.predict(input_df_survival)
                
                # Adjust prediction based on age and tumor size
                age_recode = input_data_survival["Age recode with <1 year olds"]
                tumor_size = int(input_data_survival["CS tumor size (2004-2015)"])
                
                if age_recode == "80-84 years" and tumor_size > 100:
                    prediction_survival = prediction_survival - 10  # Decrease survival months by 10
                    
                elif age_recode == "45-49 years" and tumor_size < 100:
                    prediction_survival = prediction_survival + 50  # Increase survival months by 10
                
                elif age_recode == "80-84 years" and tumor_size < 100:
                    prediction_survival = prediction_survival + 5  # Increase survival months by 5
                
                elif age_recode == "45-49 years" and tumor_size > 100:
                    prediction_survival = prediction_survival - 15  # Decrease survival months by 5
                    
                prediction_survival -= tumor_size / 5
                if prediction_survival < 0:
                    prediction_survival = abs(prediction_survival) % 15 + 1
                chemo_recode = input_data_survival["Chemotherapy recode (yes, no/unk)"]
                if chemo_recode == "No/Unknown":
                    prediction_survival = prediction_survival - 12
     
                    
                st.success(f'Predicted survival months: {prediction_survival[0]}')

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


    
        
elif page == "Treatment Response":
    st.title('Treatment Response Prediction')

    selected_features_treatment =  [
        "Age recode with <1 year olds",
        "Sex",
        "Year of diagnosis",
        "Race recode (W, B, AI, API)",
        "Site recode ICD-O-3/WHO 2008",
        "TNM 7/CS v0204+ Schema (thru 2017)",
        "Chemotherapy recode (yes, no/unk)",
        "Radiation recode",
        "RX Summ--Systemic/Sur Seq (2007+)",
        "Months from diagnosis to treatment",
        "Regional nodes examined (1988+)",
        "Regional nodes positive (1988+)",
        "CS tumor size (2004-2015)",
        "CS extension (2004-2015)",
        "CS lymph nodes (2004-2015)",
        "CS mets at dx (2004-2015)",
        "Survival months"
    ]
    target_variable_treatment = "RX Summ--Surg/Rad Seq"  # Corrected target variable
    
    X_treatment, y_treatment = preprocess_data(treatment_response, selected_features_treatment, target_variable_treatment)
    X_train_treatment, X_test_treatment, y_train_treatment, y_test_treatment = train_test_split(X_treatment, y_treatment, test_size=0.2, random_state=42)
    
    model_treatment, _ = train_random_forest(X_train_treatment, y_train_treatment)

    # User input form for treatment response prediction
    st.subheader('Enter Patient Data for Treatment Response Prediction')
    input_data_treatment = {}
    for feature in selected_features_treatment:
        input_data_treatment[feature] = st.selectbox(f'Select {feature}', options=[''] + list(treatment_response[feature].unique()))

    if st.button('Predict Treatment Response'):
        if "" in input_data_treatment.values():
            st.warning("Please fill in all fields.")
        else:
            try:
                input_df_treatment = pd.DataFrame([input_data_treatment], columns=selected_features_treatment)
                input_df_treatment = pd.get_dummies(input_df_treatment)
                missing_cols_treatment = set(X_train_treatment.columns) - set(input_df_treatment.columns)
                for col in missing_cols_treatment:
                    input_df_treatment[col] = 0
                input_df_treatment = input_df_treatment[X_train_treatment.columns]
                prediction_treatment = model_treatment.predict(input_df_treatment)
                st.success(f'Predicted treatment response: {prediction_treatment[0]}')
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


elif page == "Chemotherapy Prediction":
    st.title('Chemotherapy Prediction')

    selected_attributes_chemo = [
        'Age recode with <1 year olds',
        'Sex',
        'Race recode (W, B, AI, API)',
        'Primary Site',
        'Radiation recode',
        'RX Summ--Systemic/Sur Seq (2007+)',
        'RX Summ--Surg/Rad Seq',
        'Histologic Type ICD-O-3',
        'Months from diagnosis to treatment',
        'Grade Recode (thru 2017)',
        'Summary stage 2000 (1998-2017)',
        'Reason no cancer-directed surgery',
        'Regional nodes examined (1988+)',
        'SEER Combined Mets at DX-lung (2010+)'
    ]
    target_variable_chemo = 'Chemotherapy recode (yes, no/unk)'

    X_chemo, y_chemo = preprocess_data(chemo_data, selected_attributes_chemo, target_variable_chemo)
    X_train_chemo, X_test_chemo, y_train_chemo, y_test_chemo = train_test_split(X_chemo, y_chemo, test_size=0.2, random_state=42)

    model_chemo, _ = train_logistic_regression(X_train_chemo, y_train_chemo)

    # User input form for chemotherapy prediction
    st.subheader('Enter Patient Data for Chemotherapy Prediction')
    input_data_chemo = {}
    for feature in selected_attributes_chemo:
        input_data_chemo[feature] = st.selectbox(f'Select {feature}', options=[''] + list(chemo_data[feature].unique()))

    if st.button('Predict Chemotherapy'):
        if "" in input_data_chemo.values():
            st.warning("Please fill in all fields.")
        else:
            try:
                input_df_chemo = pd.DataFrame([input_data_chemo], columns=selected_attributes_chemo)
                input_df_chemo = pd.get_dummies(input_df_chemo)
                missing_cols_chemo = set(X_train_chemo.columns) - set(input_df_chemo.columns)
                for col in missing_cols_chemo:
                    input_df_chemo[col] = 0
                input_df_chemo = input_df_chemo[X_train_chemo.columns]
                prediction_chemo = model_chemo.predict(input_df_chemo)
                st.success(f'Predicted chemotherapy status: {prediction_chemo[0]}')
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
