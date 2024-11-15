"""
"Predict the likelihood of civil war or insurgency in a given country using different indicators?"

Intrastate War Dataset: For civil war occurrences.
National Material Capabilities (NMC) v6.0: For economic and military indicators.
Direct Contiguity v3.2: To explore effects of neighboring conflicts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load datasets (assuming CSV files, adjust if they are in other formats)
direct_continuity_df = pd.read_csv('datasets/direct-continuity.csv', encoding='latin1')
intra_state_war_df = pd.read_csv('datasets/intra-state-war.csv', encoding='latin1')
national_material_df = pd.read_csv('datasets/national-material-capabilities.csv', encoding='latin1')
country_codes_df = pd.read_csv('datasets/country-codes.csv', encoding='latin1')

# 1. Preprocessing

# Merge National Material and Country Codes to get full country names and codes
merged_df = pd.merge(national_material_df, country_codes_df, how='left', left_on='ccode', right_on='CCode')

# Convert date fields in IntraStateWar to datetime format and filter conflicts that qualify as civil war/insurgency
intra_state_war_df['StartDate1'] = pd.to_datetime(
    intra_state_war_df[['StartYear1', 'StartMonth1', 'StartDay1']].rename(columns={
        'StartYear1': 'year',
        'StartMonth1': 'month',
        'StartDay1': 'day'
    }), errors='coerce')

intra_state_war_df['EndDate1'] = pd.to_datetime(
    intra_state_war_df[['EndYear1', 'EndMonth1', 'EndDay1']].rename(columns={
        'EndYear1': 'year',
        'EndMonth1': 'month',
        'EndDay1': 'day'
    }), errors='coerce')

# Mark conflicts that are civil wars or insurgencies
intra_state_war_df['IsCivilWar'] = intra_state_war_df['WarType'].apply(
    lambda x: 1 if x in ['Civil War', 'Insurgency'] else 0)

# Create a binary target variable from intra_state_war_df for civil war prediction
civil_war_df = intra_state_war_df[['CcodeA', 'IsCivilWar', 'StartDate1', 'EndDate1']]
# 'SideADeaths', 'SideBDeaths']

# Merge with the National Material dataset to get country features
model_data = pd.merge(merged_df, civil_war_df, left_on='ccode', right_on='CcodeA', how='left')

# 2. Feature Selection
# Select relevant features for prediction
# Example features: military expenditure, population, military personnel, urban population, and deaths in war
features = ['milex', 'milper', 'tpop', 'upop', 'SideADeaths', 'SideBDeaths', 'pec', 'cinc']
target = 'IsCivilWar'

# Fill missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
model_data[features] = imputer.fit_transform(model_data[features])

# 3. Train a Random Forest Classifier

# Create feature matrix X and target vector y
X = model_data[features]
y = model_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 4. Evaluate the model
y_pred = rf_model.predict(X_test_scaled)

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
