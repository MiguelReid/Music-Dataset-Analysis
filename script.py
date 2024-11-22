"""
Can we predict the likelihood of civil war or insurgency in a given country using different indicators?

Intrastate War Dataset: For civil war occurrences.
National Material Capabilities (NMC) v6.0: For economic and military indicators.
Direct Contiguity v3.2: To explore effects of neighboring conflicts.
Country-Codes: To map country codes to full country names.

----------------------------------------------------------

V5RegionNum: 1=NorthAmerica; 2=South America; 3=Europe; 4=Sub-Saharan Africa; 5=Middle East and North Africa; 6=Asia and Oceania.
WarType: 4 = Civil war for central control; 5 = Civil war over local issues; 6 = Regional internal; 7 = Intercommunal
Intnl is the war internationalized?: 0=No; 1=Yes
OUTCOME: 1 = Side A wins; 2 = Side B wins; 3 = Compromise; 4 = Transformed into another type of war; 5 = Ongoing as of 12/31/2014;
6 = Stalemate; 7 = Conflict continues at below war level
Start/End Day,Month...For 2,3,4: When did it start after having stopped (-8 = N/A; -9 = Month Unknown)
"""
from umap import UMAP
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Country Codes --------------------------------------------
country_codes_df = pd.read_csv('datasets/country-codes.csv', encoding='latin1')
country_codes_df.drop(columns=['StateAbb'], inplace=True)
# From duplicated().sum() we see only country codes are duplicated
country_codes_df.drop_duplicates(inplace=True)
# Country Codes --------------------------------------------


# Intra War ------------------------------------------------
wars_df = pd.read_csv('datasets/intra-state-wars.csv', encoding='latin1')


# I only need the years to separate the data into non-war and war years
def drop_dates(df, prefix):
    columns_to_drop = [
        f'StartMo{prefix}', f'StartDy{prefix}',
        f'EndMo{prefix}', f'EndDy{prefix}'
    ]
    df.drop(columns=columns_to_drop, inplace=True)


# Drop the non-year dates
for i in range(1, 5):
    drop_dates(wars_df, i)

# Drop unimportant columns
wars_df.drop(columns=['WarNum', 'WarName', 'Version', 'WDuratDays', 'WDuratMo', 'TotNatMonWar', 'TransTo', 'TransFrom',
                      'DeathsSideA', 'DeathsSideB', 'TotalBDeaths'], inplace=True)

# Drop countries which do not exist anymore EGYPT NOT THE SAME? 1881 is recognized
wars_df = wars_df[wars_df['CcodeA'] != -8]

# Change Initiator values to 0, 1, 2
wars_df['Initiator'] = wars_df.apply(
    lambda row: 0 if row['Initiator'] == row['SideA']
    else 1 if row['Initiator'] == row['SideB']
    else 2, axis=1
)

wars_df.drop(columns=['SideB'], inplace=True)
# Intra War ------------------------------------------------


# Resources ------------------------------------------------
resources_df = pd.read_csv('datasets/national-material-capabilities.csv', encoding='latin1')

# From resources drop version, stateabb
resources_df.drop(columns=['version', 'stateabb'], inplace=True)
# Resources ------------------------------------------------


# Polity5 ------------------------------------------------
polity5_df = pd.read_csv('datasets/polity5.csv')
polity5_df.drop(columns=['p5', 'cyear'], inplace=True)


# Polity5 ------------------------------------------------


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

        labeled_data.append({'ccode': country, 'year': year, 'War_Occurred': is_war_year, 'milex': row['milex'],
                             'milper': row['milper'], 'irst': row['irst'], 'pec': row['pec'], 'tpop': row['tpop'],
                             'upop': row['upop'], 'cinc': row['cinc']})

    return pd.DataFrame(labeled_data)


# Apply function to label war and non-war years
labeled_df = label_war_years(wars_df, resources_df)
labeled_df = pd.merge(labeled_df, wars_df, left_on='ccode', right_on='CcodeA', how='left')

# Names for merging with POLITY5
merged_df = pd.merge(labeled_df, country_codes_df, left_on='ccode', right_on='CCode', how='left')

# Rename 'StateNme' column to 'Country'
merged_df.rename(columns={'StateNme': 'Country'}, inplace=True)

# Merge the labeled data with polity5
merged_df = pd.merge(merged_df, polity5_df, left_on=['Country', 'year'], right_on=['country', 'year'],
                     how='left')

# Drop every redundant, repeated and unnecessary column
merged_df.drop(columns=['ccode_x', 'ccode_y', 'CCode', 'Country'], inplace=True)

# Some names won't be the same due to different naming conventions so we'll get rid of them
merged_df = merged_df[merged_df['country'].notnull()]

# Save merged_df to a CSV file
merged_df.to_csv('datasets/merged-data.csv', index=False)

# ML MODEL --------------------------------------------

# Select features and target
features = [
    'milex', 'milper', 'irst', 'pec', 'tpop', 'upop', 'cinc',  # Material capabilities
    'WarType', 'Intnl', 'Outcome', 'V5RegionNum', # From wars dataset
    'polity', 'polity2', 'durable', 'xrreg', 'xrcomp', 'xropen', 'polcomp',  # From Polity5 dataset
]
target = 'War_Occurred'

# Split into training and testing data
X = merged_df[features]
y = merged_df[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# 80% train 20% test

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('---------------------------')
print(classification_report(y_test, y_pred))

# Predict war probability for the latest available data
last_year_df = merged_df.loc[merged_df.groupby('country')['year'].idxmax()].reset_index(drop=True)
latest_data_scaled = scaler.transform(last_year_df[features])
last_year_df['WarProbability'] = model.predict_proba(latest_data_scaled)[:, 1]

print(last_year_df.columns)

# Display the updated latest_data with country names
print(last_year_df[['country', 'WarProbability']].to_string())

# PLOTTING -----------------------------

# Get feature importances from the model
importances = model.feature_importances_
feature_names = features

# Create a DataFrame for plotting
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')


# Check for NaN values in the DataFrame
nan_counts = merged_df.isna().sum()

# Filter columns that have NaN values
nan_columns = nan_counts[nan_counts > 0]

# Print the columns with NaN values and their counts
print(nan_columns)
"""


# Fit and transform the scaled features using UMAP
umap_model = UMAP(n_neighbors=15, n_components=2, random_state=42, n_jobs=1)
X_umap = umap_model.fit_transform(X_scaled)

# Plot the UMAP projection
plt.figure(figsize=(10, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='coolwarm', s=5)
plt.title('UMAP Projection of Features')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='War Occurred')
"""
plt.show()
