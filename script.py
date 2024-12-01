"""
Can we predict the likelihood of civil war or insurgency in a given country using different indicators?

Intrastate War Dataset: For civil war occurrences.
National Material Capabilities (NMC) v6.0: For economic and military indicators.
Direct Contiguity v3.2: To explore effects of neighboring conflicts.
Country-Codes: To map country codes to full country names.
Religion: To investigate the role of religion in conflict.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

country_codes_df = pd.read_csv('datasets/country-codes.csv', encoding='latin1')
wars_df = pd.read_csv('datasets/intra-state-wars.csv', encoding='latin1')
polity5_df = pd.read_csv('datasets/polity5.csv')
resources_df = pd.read_csv('datasets/national-material-capabilities.csv', encoding='latin1')
religion_df = pd.read_csv('datasets/religion.csv', encoding='latin1')

# Limit every dataset to the years 1818-2014 (Range of intra-state-wars)
polity5_df = polity5_df[(polity5_df['year'] >= 1945) & (polity5_df['year'] <= 2010)]
resources_df = resources_df[(resources_df['year'] >= 1945) & (resources_df['year'] <= 2010)]

# Country Codes --------------------------------------------
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
# Polity5 ------------------------------------------------


# Religion -----------------------------------------------
religion_df.drop(columns=['Version', 'sourcecode', 'datatype', 'dualrelig', 'total'])
columns_to_replace = ['chrstprot', 'chrstcat', 'chrstorth', 'chrstang', 'chrstothr', 'chrstgen', 'judorth', 'jdcons',
                      'judref', 'judothr', 'judgen', 'islmsun', 'islmshi', 'islmibd', 'islmnat', 'islmalw', 'islmahm',
                      'islmothr', 'islmgen', 'budmah', 'budthr', 'budothr', 'budgen', 'zorogen', 'hindgen', 'sikhgen',
                      'shntgen', 'bahgen', 'taogen', 'jaingen', 'confgen', 'syncgen', 'anmgen', 'nonrelig', 'othrgen',
                      'sumrelig', 'pop', 'chrstprotpct', 'chrstcatpct', 'chrstorthpct', 'chrstangpct', 'chrstothrpct',
                      'chrstgenpct', 'judorthpct', 'judconspct', 'judrefpct', 'judothrpct', 'judgenpct', 'islmsunpct',
                      'islmshipct', 'islmibdpct', 'islmnatpct', 'islmalwpct', 'islmahmpct', 'islmothrpct', 'islmgenpct',
                      'budmahpct', 'budthrpct', 'budothrpct', 'budgenpct', 'zorogenpct', 'hindgenpct', 'sikhgenpct',
                      'shntgenpct', 'bahgenpct', 'taogenpct', 'jaingenpct', 'confgenpct', 'syncgenpct', 'anmgenpct',
                      'nonreligpct', 'othrgenpct', 'sumreligpct']

for column in columns_to_replace:
    religion_df[column] = religion_df.groupby('name')[column].transform(lambda x: x.replace(0, x.mean()))


# Religion -----------------------------------------------


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
                wars.iloc[0].to_dict()
                break

        labeled_row = {
            'ccode': country,
            'year': year,
            'War_Occurred': is_war_year,
            'milex': row['milex'],
            'milper': row['milper'],
            'irst': row['irst'],
            'pec': row['pec'],
            'tpop': row['tpop'],
            'upop': row['upop'],
            'cinc': row['cinc'],
            'V5RegionNum': wars['V5RegionNum'].iloc[0] if not wars.empty else pd.NA
        }

        labeled_data.append(labeled_row)

    return pd.DataFrame(labeled_data)


# Label war and non-war years
labeled_df = label_war_years(wars_df, resources_df)

# Names for merging with POLITY5
merged_df = pd.merge(labeled_df, country_codes_df, left_on='ccode', right_on='CCode', how='inner')

merged_df = pd.merge(merged_df, religion_df, left_on=['StateAbb', 'year'], right_on=['name', 'year'], how='inner')

# Merge the labeled data with polity5
merged_df = pd.merge(merged_df, polity5_df, left_on=['StateNme', 'year'], right_on=['country', 'year'],
                     how='inner')

# Drop every redundant, repeated and unnecessary column
merged_df.drop(columns=['ccode_x', 'ccode_y', 'StateNme', 'scode', 'flag'], inplace=True)

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

# List of religion percentage columns
religion_columns = [
    'chrstprotpct', 'chrstcatpct', 'chrstorthpct', 'chrstangpct', 'chrstothrpct', 'chrstgenpct',
    'judorthpct', 'judconspct', 'judrefpct', 'judothrpct', 'judgenpct',
    'islmsunpct', 'islmshipct', 'islmibdpct', 'islmnatpct', 'islmalwpct', 'islmahmpct', 'islmothrpct', 'islmgenpct',
    'budmahpct', 'budthrpct', 'budothrpct', 'budgenpct',
    'zorogenpct', 'hindgenpct', 'sikhgenpct', 'shntgenpct', 'bahgenpct', 'taogenpct', 'jaingenpct', 'confgenpct',
    'syncgenpct', 'anmgenpct', 'nonreligpct', 'othrgenpct', 'sumreligpct'
]


# Check if a dominant religion (lower entropy) is more peaceful than those with greater religious diversity
def calculate_entropy(row):
    probabilities = row[religion_columns].values
    probabilities = probabilities[probabilities > 0]  # Remove zero values
    probabilities = np.array(probabilities, dtype=np.float64)  # Ensure numpy array type
    return -np.sum(probabilities * np.log(probabilities))


# Calculate entropy for each row
merged_df['religion_entropy'] = merged_df.apply(calculate_entropy, axis=1)

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
    'V5RegionNum',  # Intra-War-Dataset
    'religion_entropy',
    # Religious Features
    'chrstprot', 'chrstcat', 'chrstorth', 'chrstang', 'chrstothr', 'chrstgen', 'judorth', 'jdcons', 'judref', 'judothr',
    'judgen', 'islmsun', 'islmshi', 'islmibd', 'islmnat', 'islmalw', 'islmahm', 'islmothr', 'islmgen', 'budmah',
    'budthr', 'budothr', 'budgen', 'zorogen', 'hindgen', 'sikhgen', 'shntgen', 'bahgen', 'taogen', 'jaingen', 'confgen',
    'syncgen', 'anmgen', 'nonrelig', 'othrgen', 'sumrelig', 'pop', 'chrstprotpct', 'chrstcatpct', 'chrstorthpct',
    'chrstangpct', 'chrstothrpct', 'chrstgenpct', 'judorthpct', 'judconspct', 'judrefpct', 'judothrpct', 'judgenpct',
    'islmsunpct', 'islmshipct', 'islmibdpct', 'islmnatpct', 'islmalwpct', 'islmahmpct', 'islmothrpct', 'islmgenpct',
    'budmahpct', 'budthrpct', 'budothrpct', 'budgenpct', 'zorogenpct', 'hindgenpct', 'sikhgenpct', 'shntgenpct',
    'bahgenpct', 'taogenpct', 'jaingenpct', 'confgenpct', 'syncgenpct', 'anmgenpct', 'nonreligpct', 'othrgenpct',
    'sumreligpct'

]
target = 'War_Occurred'

# Train-test split by year
train_data = merged_df[merged_df['year'] < 1997]
test_data = merged_df[merged_df['year'] >= 1997]
# 80% of the data is used for training

X_train = train_data[features].dropna()
y_train = train_data[target].loc[X_train.index]

X_test = test_data[features].dropna()
y_test = test_data[target].loc[X_test.index]

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample')
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on test data
y_pred = model.predict(X_test_pca)
y_pred_proba = model.predict_proba(X_test_pca)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba)}")
# Ensure features are properly prepared for the entire dataset
X_all = merged_df[features].dropna()  # Drop rows with missing feature values

# Scale the features using the trained scaler
X_all_scaled = scaler.transform(X_all)

# Apply PCA to the scaled features
X_all_pca = pca.transform(X_all_scaled)

# Predict WarProbability for all rows
merged_df.loc[X_all.index, 'WarProbability'] = model.predict_proba(X_all_pca)[:, 1]

# Get only the last year of data in merged_df
last_year_df = merged_df[merged_df['year'] == 2010]

# Print the country, year, and predicted percentage of WarProbability over 0.0 and sort by WarProbability
print(last_year_df[last_year_df['WarProbability'] > 0.0].sort_values(by='WarProbability', ascending=False)[
          ['country', 'year', 'WarProbability']].to_string())

# Dataframe Information
print(f"Number of data points: {merged_df.shape[0]}")
print(f"Number of features: {merged_df.shape[1]}")

# PLOTTING --------------------------------------------

# Calculate the ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.tight_layout()
plt.show()

filtered_features = [
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
    'V5RegionNum',  # Intra-War-Dataset
    'religion_entropy'
]

# Calculate the correlation matrix
X_all_selected = X_all[filtered_features]
# Calculate the correlation matrix for the selected features
correlation_matrix = X_all_selected.corr()
# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Get the feature importances from the model
feature_importances = model.feature_importances_
# Get the names of the PCA components
# Get the original feature names
original_feature_names = X_train.columns

# Get the PCA components
pca_components = pca.components_

# Get the feature names for each principal component
pca_feature_names = []
for component in pca_components:
    # Get the index of the feature with the highest absolute value in the component
    feature_index = np.argmax(np.abs(component))
    # Get the corresponding feature name
    pca_feature_names.append(original_feature_names[feature_index])

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    'Feature': pca_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot the top 15 feature importances
top_features = feature_importances_df.head(15)
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

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
plt.tight_layout()
plt.show()

# Filter data for analysis
analysis_data = merged_df[
    ['tpop', 'polity2', 'WarProbability', 'parreg', 'islmgen', 'religion_entropy']]

filtered_data = analysis_data[analysis_data['WarProbability'] > 0.00]  # Exclude near-zero values

# Scatter plot for 'tpop' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=filtered_data, x='tpop', y='WarProbability', alpha=0.7, color='blue', s=20)
sns.regplot(data=filtered_data, x='tpop', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Total Population (tpop) and War Probability')
plt.xlabel('Total Population (tpop)')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()

# Scatter plot for 'polity2' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='polity2', y='WarProbability', alpha=0.7, color='green', s=20)
sns.regplot(data=filtered_data, x='polity2', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Polity and War Probability')
plt.xlabel('Polity')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()

# Scatter plot for 'islmgen' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='islmgen', y='WarProbability', alpha=0.7, color='green', s=20)
sns.regplot(data=filtered_data, x='islmgen', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Relationship Between Islmgen and War Probability')
plt.xlabel('Islmgen')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()

# Scatter plot for 'parreg' vs. 'WarProbability'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='parreg', y='WarProbability', alpha=0.7, color='blue', s=20)
sns.regplot(data=filtered_data, x='parreg', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Parreg and War Probability')
plt.xlabel('Parreg (Regulation of Participation)')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()

# Calculate the average WarProbability for each year
avg_war_prob_by_year = merged_df.groupby('year')['WarProbability'].mean().reset_index()
# Plot the line chart
plt.figure(figsize=(12, 6))
plt.plot(avg_war_prob_by_year['year'], avg_war_prob_by_year['WarProbability'], marker='o', linestyle='-', color='b')
plt.title('Trend of Average War Probability Over the Years')
plt.xlabel('Year')
plt.ylabel('Average War Probability')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the correlation between religion entropy and WarProbability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='religion_entropy', y='WarProbability', alpha=0.7, s=20)
sns.regplot(data=filtered_data, x='religion_entropy', y='WarProbability', scatter=False, color='red', ci=None)
plt.title('Correlation Between Religion Entropy and War Probability')
plt.xlabel('Religion Entropy')
plt.ylabel('War Probability (%)')
plt.tight_layout()
plt.show()

# Pairplot
"""
sns.pairplot(merged_df[filtered_features + [target]], diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.tight_layout()
plt.show()
"""
