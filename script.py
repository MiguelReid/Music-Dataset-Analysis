"""
Can we predict the likelihood of civil war or insurgency in a given country using different indicators?

Intrastate War Dataset: For civil war occurrences.
National Material Capabilities (NMC) v6.0: For economic and military indicators.
Direct Contiguity v3.2: To explore effects of neighboring conflicts.
Country-Codes: To map country codes to full country names.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
wars_df = pd.read_csv('datasets/intra-state-wars.csv', encoding='latin1')

""" V5RegionNum: 1=NorthAmerica; 2=South America; 3=Europe; 4=Sub-Saharan Africa; 5=Middle East and North Africa; 6=Asia and Oceania.

WarType: 4 = Civil war for central control; 5 = Civil war over local issues; 6 = Regional internal; 7 = Intercommunal

Intnl is the war internationalized?: 0=No; 1=Yes 

OUTCOME: 1 = Side A wins; 2 = Side B wins; 3 = Compromise; 4 = Transformed into another type of war; 5 = Ongoing as of 12/31/2014; 
6 = Stalemate; 7 = Conflict continues at below war level 

Start/End Day,Month...For 2,3,4: When did it start after having stopped (-8 = N/A; -9 = Month Unknown) """

resources_df = pd.read_csv('datasets/national-material-capabilities.csv', encoding='latin1')
country_codes_df = pd.read_csv('datasets/country-codes.csv', encoding='latin1')

# 2. Remove Duplicates
wars_df.drop_duplicates(inplace=True)
resources_df.drop_duplicates(inplace=True)
country_codes_df.drop_duplicates(inplace=True)

# Drop rows whose CcodeA = -8 (Countries which do not exist anymore) EGYPT NOT THE SAME? 1881 is recognized
wars_df = wars_df[wars_df['CcodeA'] != -8]

# Change Initiator values to 0, 1, 2
wars_df['Initiator'] = wars_df.apply(
    lambda row: 0 if row['Initiator'] == row['SideA']
    else 1 if row['Initiator'] == row['SideB']
    else 2, axis=1
)

# Combine year, month, and day into a single date column
def combine_date_columns(df, prefix):
    # I only need the years to separate the data into non war and war years
    columns_to_drop = [
        f'StartMo{prefix}', f'StartDy{prefix}',
        f'EndMo{prefix}', f'EndDy{prefix}'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

# Loop through different indexes
for i in range(1, 5):
    combine_date_columns(wars_df, i)

# Function to generate a war-year indicator for each country
def label_war_years(war_df, df):
    labeled_data = []

    # Iterate through national material capabilities (country, year pairs)
    for _, row in df.iterrows():
        country = row['ccode']
        year = row['year']

        # Check if the year falls within any war range for that country
        wars = war_df[war_df['CcodeA'] == country]
        is_war_year = 0  # Default to non-war

        for i in range(1, 5):  # Check up to 4 war periods
            if any((wars[f'StartYr{i}'] <= year) & (year <= wars[f'EndYr{i}'])):
                is_war_year = 1
                break

        labeled_data.append({'ccode': country, 'year': year, 'War_Occurred': is_war_year})

    return pd.DataFrame(labeled_data)

# Apply function to label war and non-war years
labeled_df = label_war_years(wars_df, resources_df)

# Merge the labeled data with national material capabilities
merged_df = pd.merge(labeled_df, resources_df, on=['ccode', 'year'])

# --- New Step: Get Last Year of Data for Each Country ---
last_year_df = resources_df.loc[resources_df.groupby('ccode')['year'].idxmax()].reset_index(drop=True)

# Select features and target
features = ['milex', 'milper', 'irst', 'pec', 'tpop', 'upop', 'cinc']
target = 'War_Occurred'

# Split into training and testing data
X = merged_df[features]
y = merged_df[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict war probability for the latest available data
latest_data = resources_df.loc[resources_df.groupby('ccode')['year'].idxmax()].reset_index(drop=True)
latest_data_scaled = scaler.transform(latest_data[features])
latest_data['War_Probability'] = model.predict_proba(latest_data_scaled)[:, 1]

# Merge latest_data with country_codes_df to get country names
latest_data = pd.merge(latest_data, country_codes_df[['CCode', 'StateNme']], left_on='ccode', right_on='CCode', how='left')

# Drop the 'ccode' and 'CCode' columns as they are no longer needed
latest_data.drop(columns=['ccode', 'CCode'], inplace=True)

# Rename 'StateNme' column to 'Country'
latest_data.rename(columns={'StateNme': 'Country'}, inplace=True)

# Display the updated latest_data with country names
print(latest_data[['Country', 'War_Probability']].to_string())
