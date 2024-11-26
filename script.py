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
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

country_codes_df = pd.read_csv('datasets/country-codes.csv', encoding='latin1')
wars_df = pd.read_csv('datasets/intra-state-wars.csv', encoding='latin1')
polity5_df = pd.read_csv('datasets/polity5.csv')
resources_df = pd.read_csv('datasets/national-material-capabilities.csv', encoding='latin1')

# Limit every dataset to the years 1818-2014 (Range of intra-state-wars)
polity5_df = polity5_df[(polity5_df['year'] >= 1818) & (polity5_df['year'] <= 2014)]
resources_df = resources_df[(resources_df['year'] >= 1818) & (resources_df['year'] <= 2014)]

# Country Codes --------------------------------------------
country_codes_df.drop(columns=['StateAbb'], inplace=True)
# From duplicated().sum() we see only country codes are duplicated
country_codes_df.drop_duplicates(inplace=True)


# Country Codes --------------------------------------------


# Intra War ------------------------------------------------
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

# Print how many -8 or -9 values does each column have
# print('CHECK -> ',wars_df.isin([-8, -9]).sum())


# Intra War ------------------------------------------------


# Resources ------------------------------------------------
# From resources drop version, stateabb
resources_df.drop(columns=['version', 'stateabb'], inplace=True)

# Replace every -9 value with the mean of their column
columns_to_replace = [
    'milex', 'milper', 'irst', 'pec', 'tpop', 'upop', 'cinc'
]

for column in columns_to_replace:
    resources_df[column] = resources_df.groupby('ccode')[column].transform(lambda x: x.replace(-9, x.mean()))

# Resources ------------------------------------------------


# Polity5 ------------------------------------------------
# Drop these columns with high number of NaN values
polity5_df.drop(columns=['fragment', 'prior', 'emonth', 'eday', 'eyear', 'eprec', 'interim', 'bmonth', 'bday', 'byear',
                         'bprec', 'post', 'change', 'd5', 'sf', 'regtrans', 'p5', 'cyear'], inplace=True)

# Correlates of war only has records until 2016
polity5_df = polity5_df[polity5_df['year'] <= 2016]


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


# Label war and non-war years
labeled_df = label_war_years(wars_df, resources_df)
# Changed to inner join
labeled_df = pd.merge(labeled_df, wars_df[['CcodeA', 'WarType', 'Intnl', 'Outcome', 'V5RegionNum']], left_on='ccode',
                      right_on='CcodeA', how='inner')

labeled_df.drop(columns=['CcodeA'], inplace=True)

# Names for merging with POLITY5
merged_df = pd.merge(labeled_df, country_codes_df, left_on='ccode', right_on='CCode', how='left')

# Merge the labeled data with polity5
merged_df = pd.merge(merged_df, polity5_df, left_on=['StateNme', 'year'], right_on=['country', 'year'],
                     how='left')

# Drop every redundant, repeated and unnecessary column
merged_df.drop(columns=['ccode_x', 'ccode_y', 'StateNme', 'scode'], inplace=True)

# Some names won't be the same due to different naming conventions so we'll get rid of them
merged_df = merged_df[merged_df['country'].notnull()]

# Replace missing values with NaN
merged_df.replace([-66, -77, -88], pd.NA, inplace=True)
columns_to_replace = [
    'democ', 'autoc', 'polity2', 'durable', 'xrreg', 'xrcomp',
    'xropen', 'xconst', 'parreg', 'parcomp', 'exrec', 'exconst', 'polcomp'
]

for column in columns_to_replace:
    merged_df[column] = merged_df.groupby('CCode')[column].transform(lambda x: x.fillna(x.mean()))
# Save the updated DataFrame to a CSV file
merged_df.to_csv('datasets/merged-data.csv', index=False)

# ML MODEL --------------------------------------------

# Select lagged features and drop war-related features
features = [
    'polity2',  # Indicators of political regime type
    'democ', 'autoc',  # Democracy and autocracy score
    'durable',  # Uninterrupted years of stability
    'xrreg', 'xrcomp', 'xropen',  # External regime features
    'polcomp',  # Political competition
    'parreg',  # Party regulation
    'milex',  # Military expenditures
    'milper',  # Military personnel
    'irst',  # Industrial production
    'pec',  # Primary energy consumption
    'tpop',  # Total population
    'upop',  # Urban population
    'cinc',  # Composite Index of National Capabilities
    'V5RegionNum'  # Intra-War-Dataset
]
target = 'War_Occurred'

# Train-test split by year
train_data = merged_df[merged_df['year'] < 2010]
test_data = merged_df[merged_df['year'] >= 2010]

X_train = train_data[features].dropna()
y_train = train_data[target].loc[X_train.index]

X_test = test_data[features].dropna()
y_test = test_data[target].loc[X_test.index]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train model with balanced weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train_resampled)

# Evaluate on test data
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

# Predict war probability for the latest available data
last_year_df = merged_df.loc[merged_df.groupby('country')['year'].idxmax()].reset_index(drop=True)
latest_data_scaled = scaler.transform(last_year_df[features])
last_year_df['WarProbability'] = model.predict_proba(latest_data_scaled)[:, 1]

# Print the country, year, and predicted percentage of WarProbability over 0.0 and sort by WarProbability
print(last_year_df[last_year_df['WarProbability'] > 0.0].sort_values(by='WarProbability', ascending=False)[
          ['country', 'year', 'WarProbability']].to_string())

# PLOTTING --------------------------------------------

"""
# Select only numeric columns for the correlation matrix
numeric_columns = merged_df.select_dtypes(include=['number'])

# Calculate the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance')

# Mapping of V5RegionNum to region names
region_mapping = {
    1: 'North America',
    2: 'South America',
    3: 'Europe',
    4: 'Sub-Saharan Africa',
    5: 'Middle East and North Africa',
    6: 'Asia and Oceania'
}

# Calculate the average war probability for each V5RegionNum
avg_war_prob_by_region = last_year_df.groupby('V5RegionNum')['WarProbability'].mean().reset_index()

# Replace V5RegionNum with region names
avg_war_prob_by_region['V5RegionNum'] = avg_war_prob_by_region['V5RegionNum'].map(region_mapping)

# Pivot the data for the heatmap
heatmap_data = avg_war_prob_by_region.pivot_table(index='V5RegionNum', values='WarProbability')

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Average War Probability (%)'})
plt.title('Average War Probability by Region')
plt.xlabel('Region')
plt.ylabel('Region')

# Filter data for analysis
analysis_data = last_year_df[['milex', 'pec', 'cinc', 'tpop', 'polity2', 'WarProbability']]

# Plot 1: Scatter plot for 'tpop' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_data, x='tpop', y='WarProbability', alpha=0.7, color='blue', s=20)
sns.regplot(data=analysis_data, x='tpop', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Total Population (tpop) and War Probability')
plt.xlabel('Total Population (tpop)')
plt.ylabel('War Probability (%)')
plt.tight_layout()

# Plot 2: Scatter plot for 'polity2' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_data, x='polity2', y='WarProbability', alpha=0.7, color='green', s=20)
sns.regplot(data=analysis_data, x='polity2', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Polity and War Probability')
plt.xlabel('Polity')
plt.ylabel('War Probability (%)')
plt.tight_layout()

# Plot 2: Scatter plot for 'cinc' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_data, x='cinc', y='WarProbability', alpha=0.7, color='green', s=20)
sns.regplot(data=analysis_data, x='cinc', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Cinc and War Probability')
plt.xlabel('The Composite Index of National Capability (cinc)')
plt.ylabel('War Probability (%)')
plt.tight_layout()

# Plot 1: Scatter plot for 'milex' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_data, x='milex', y='WarProbability', alpha=0.7, color='blue', s=20)
sns.regplot(data=analysis_data, x='milex', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Military Expenditures (milex) and War Probability')
plt.xlabel('Military Expenditures (milex)')
plt.ylabel('War Probability (%)')
plt.tight_layout()

# Plot 2: Scatter plot for 'pec' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=analysis_data, x='pec', y='WarProbability', alpha=0.7, color='green', s=20)
sns.regplot(data=analysis_data, x='pec', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Primary Energy Consumption (pec) and War Probability')
plt.xlabel('Primary Energy Consumption (pec)')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()
"""