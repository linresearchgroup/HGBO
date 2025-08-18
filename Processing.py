import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

data=pd.read_excel("PFAS soil Database.xlsx", nrows=52)
data.head()
##########################Data Preprocessing
data = data.dropna(axis=1, how='all')

numeric_cols = data.select_dtypes(include=['number']).columns
non_numeric_cols = data.select_dtypes(exclude=['number']).columns

# fill numeric columns with mean
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# fill non-numeric columns with 'Unkown'
data[non_numeric_cols] = data[non_numeric_cols].fillna('Unknown')
data = data.loc[:, data.nunique() > 1]

selected_columns = ['Material Name', 'Weight_ratio', 'Initial total mass', 'Init. Res.(Ω)']
data = data[selected_columns]

data_selected = data

# Step 3: OneHot Encoding for categorical features (keeping original columns for later use)
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = one_hot_encoder.fit_transform(data_selected[['Material Name', 'Weight_ratio']])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(['Material Name', 'Weight_ratio']))

# Merge encoded features with the original data
data_encoded = pd.concat([data_selected.reset_index(drop=True), encoded_df], axis=1)
data_encoded = data_encoded.drop(columns=['Material Name', 'Weight_ratio'])

# Step 4: Scale the numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_encoded[['Initial total mass', 'Init. Res.(Ω)'] + list(encoded_df.columns)])

# Step 5: Apply Local Outlier Factor (LOF) to identify outliers
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outlier_flags_lof = lof.fit_predict(scaled_data)
data_encoded['Outlier_LOF'] = (outlier_flags_lof == -1).astype(int)  # Mark outliers as 1

# Step 6: Group-Based Outlier Detection
grouped = data_selected.groupby(['Material Name', 'Weight_ratio', 'Initial total mass'])
data_encoded['Outlier_Group'] = 0

for name, group in grouped:
    if len(group) > 1:  # Only consider groups with more than one element
        mean_res = group['Init. Res.(Ω)'].mean()
        std_res = group['Init. Res.(Ω)'].std()
        # Mark rows that deviate significantly from the mean (e.g., > 2 standard deviations)
        outliers_in_group = group[np.abs(group['Init. Res.(Ω)'] - mean_res) > 2 * std_res].index
        data_encoded.loc[outliers_in_group, 'Outlier_Group'] = 2

# Step 7: Combine Outlier Labels
data_encoded['Outlier_Final'] = data_encoded[['Outlier_LOF', 'Outlier_Group']].max(axis=1)
data_encoded.loc[(data_encoded['Outlier_LOF'] == 1) & (data_encoded['Outlier_Group'] == 2), 'Outlier_Final'] = 3

# Step 8: Apply PCA to reduce the dimensionality to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Create a DataFrame with the PCA results and outlier labels
pca_df = pd.DataFrame(pca_result, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Outlier_Final'] = data_encoded['Outlier_Final']

outliers_labeled_1 = data_encoded[data_encoded['Outlier_Final'] == 1]
outliers_labeled_2 = data_encoded[data_encoded['Outlier_Final'] == 2]
outliers_labeled_3 = data_encoded[data_encoded['Outlier_Final'] == 3]

data_encoded['Material Name'] = data_selected['Material Name']
data_encoded['Weight_ratio'] = data_selected['Weight_ratio']
data_encoded['Initial total mass'] = data_selected['Initial total mass']

grouped_2 = data_encoded[data_encoded['Outlier_Final'] == 2].groupby(['Material Name', 'Weight_ratio', 'Initial total mass'])
for name, group in grouped_2:
    mean_value = group['Init. Res.(Ω)'].mean()
    data_encoded.loc[group.index, 'Init. Res.(Ω)'] = mean_value

# Remove rows labeled as 3
data_encoded = data_encoded[data_encoded['Outlier_Final'] != 3]

# Remove the first row labeled as 1
#outliers_labeled_1 = data_encoded[data_encoded['Outlier_Final'] == 1]
#if not outliers_labeled_1.empty:
    #data_encoded = data_encoded.drop(outliers_labeled_1.index[0])

data_encoded = data_encoded[data_encoded['Outlier_Final'] != 3]

# Keep cleaned data as cdata
cdata = data_encoded[['Material Name', 'Weight_ratio', 'Initial total mass', 'Init. Res.(Ω)']].copy()