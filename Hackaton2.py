import kagglehub
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import skew, kurtosis

# %% [markdown]
# **Price Trackers ✨**
# Margo Tiamanova
# Florencia Ogorinsky
# ***Price Comparison Tool for Airbnb***

# %% [markdown]
# **Data cleaning and preprocessing**

# %%
path = kagglehub.dataset_download("arianazmoudeh/airbnbopendata")

print("Path to dataset files:", path)

csv_file = None
for filename in os.listdir(path):
    if filename.endswith(".csv"):
        csv_file = os.path.join(path, filename)
        break

data = pd.read_csv(csv_file)

data.info()

# %%
data = data.sample(frac=0.05, random_state=42)

print(f"Sample DataFrame shape: {data.shape}")

# %%
data.head()

# %% [markdown]
# **Handling missing values**

# %%
missing_data = data.isnull().sum()
print(missing_data)

# %% [markdown]
# **Target variable: price**

# %% [markdown]
# **Dropping columns with very high missing values**

# %%
data = data.drop(['house_rules', 'license'], axis=1)

# %% [markdown]
# **Dropping rows with missing latitude and longitude (they are just a few)**

# %%
data = data.dropna(subset=['lat', 'long'])

# %%
data.columns

# %% [markdown]
# **Impute numerical columns, separating currency into a new column. If there are no reviews then number of reviews is filled with 0. We separate currency in a new column so we can analyze prices as numerical values**

# %%
def extract_currency_and_value(price_str):
    if isinstance(price_str, str):
        currency = ''.join(filter(lambda x: not x.isdigit() and x != '.' and x != ',', price_str)).strip()
        value = ''.join(filter(lambda x: x.isdigit() or x == '.' or x == ',', price_str)).strip()
        return currency, value
    else:
        return '', np.nan

data[['price_currency', 'price_value']] = data['price'].apply(extract_currency_and_value).apply(pd.Series)
data[['service_fee_currency', 'service_fee_value']] = data['service fee'].apply(extract_currency_and_value).apply(pd.Series)

data['price_value'] = pd.to_numeric(data['price_value'].str.replace(',', '', regex=False), errors='coerce')
data['service_fee_value'] = pd.to_numeric(data['service_fee_value'].str.replace(',', '', regex=False), errors='coerce')

data = data.drop(['price', 'service fee'], axis=1)

numerical_cols = ['price_value', 'service_fee_value', 'minimum nights', 'number of reviews', 'review rate number', 'calculated host listings count', 'availability 365', 'Construction year']
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())

categorical_cols = ['NAME', 'host_identity_verified', 'host name', 'neighbourhood group', 'neighbourhood', 'country', 'country code', 'instant_bookable', 'cancellation_policy']
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data['last review'] = pd.to_datetime(data['last review'], errors='coerce')
data['last review'] = data['last review'].fillna(pd.NaT)
data['reviews per month'] = data['reviews per month'].fillna(0)

print(data.isnull().sum())
data.head()

# %% [markdown]
# **Removing future dates if we have, because is not logic**

# %%
today = pd.to_datetime('today')
future_dates = data['last review'] > today
data.loc[future_dates, 'last review'] = pd.NaT
data['last review'] = data['last review'].fillna(pd.NaT)
print(data.isnull().sum())

# %% [markdown]
# **We still have some empty values in the dates but we will leave it for now**

# %% [markdown]
# **Removing duplicate rows**

# %%
data = data.drop_duplicates()
print(data.duplicated().sum())

# %% [markdown]
# **Convert boolean values to integers**

# %%
data['instant_bookable'] = data['instant_bookable'].map({True: 1, False: 0})

# %%
pd.set_option('display.max_columns', None)
data.head()

# %%
print(data.select_dtypes(include=['object']).columns)

# %% [markdown]
# Servise fee including in price, so we can sum them

# %%
data['total_price'] = data['price_value'] + data['service_fee_value']
data.drop(['price_value', 'service_fee_value'], axis=1, inplace=True)
data.head()

# %%
data.columns

# %% [markdown]
# **Transforming categorical columns with one hot encoding**

# %%
categorical_cols = ['host_identity_verified', 'neighbourhood group', 'cancellation_policy', 'room type']
for col in categorical_cols:
    dummies = pd.get_dummies(data[col], prefix=col, dummy_na=False)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
data.head()

# %%
numerical_cols = ['total_price', 'minimum nights', 'number of reviews', 'calculated host listings count', 'availability 365', 'reviews per month']
skewness = data[numerical_cols].skew()
kurtosis_values = data[numerical_cols].kurtosis()
print("Skewness:\n", skewness)
print("\nKurtosis:\n", kurtosis_values)

# %% [markdown]
# **deleting outliers and normalizing**

# %%
numerical_cols = ['minimum nights', 'number of reviews', 'calculated host listings count', 'reviews per month']

def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

for col in ['minimum nights', 'number of reviews', 'calculated host listings count', 'reviews per month']:
    data = cap_outliers_iqr(data, col)

skewed_cols = ['minimum nights', 'number of reviews', 'calculated host listings count', 'reviews per month']
normal_cols = ['total_price','availability 365']

scaler_minmax = MinMaxScaler()
data[skewed_cols] = scaler_minmax.fit_transform(data[skewed_cols])

scaler_standard = StandardScaler()
data[normal_cols] = scaler_standard.fit_transform(data[normal_cols])

data.head()

# %%
data.info()

# %%
data.columns

# %%
print(data.dtypes)

boolean_columns = data.select_dtypes(include=['bool']).columns

print(data[boolean_columns])



# %%
# Convert boolean columns to integers (0 or 1)
for col in boolean_columns:
    data[col] = data[col].astype(int)

# Verify the conversion
print(data.dtypes)  # Check the data types of the columns

# %% [markdown]
# Convert boolean vlues in 0 and 1

# %%
# Correlations ordered by absolute value

numerical_cols = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numerical_cols.corr()

price_correlation = correlation_matrix['total_price'].drop('total_price', errors='ignore')

# Order by absolute value
absolute_correlations = price_correlation.abs().sort_values(ascending=False)

# Print the 10 strongest correlations
print("Top 10 Correlations (Absolute Value):")
for col in absolute_correlations.head(10).index:
    correlation = price_correlation[col]
    sign = "Positive" if correlation > 0 else "Negative"
    print(f"{col}: {correlation:.4f} ({sign})")

# %%
# Correlations by subgroups

# Correlations by subgroups (using one-hot encoded columns)

# Get one-hot encoded columns related to 'neighbourhood group'
neighbourhood_group_cols = [col for col in data.columns if col.startswith('neighbourhood group_')]

# Check if any one-hot encoded columns exist
if neighbourhood_group_cols:
    # Iterate through one-hot encoded columns
    for col in neighbourhood_group_cols:
        group_name = col.replace('neighbourhood group_', '')
        
        # Create a subgroup based on the one-hot encoded column
        group_data = data[data[col] == 1]

        numerical_group = group_data.select_dtypes(include=['float64', 'int64'])
        group_correlation = numerical_group.corr()['total_price'].drop('total_price', errors='ignore')
        strongest_group_corr = group_correlation.abs().sort_values(ascending=False).head(5)

        if not strongest_group_corr.empty:
            print(f"\nTop Correlations in {group_name}:")
            for corr_col in strongest_group_corr.index:
                corr = group_correlation[corr_col]
                sign = "Positive" if corr > 0 else "Negative"
                print(f"  {corr_col}: {corr:.4f} ({sign})")
else:
    print("No one-hot encoded columns for 'neighbourhood group' found. Skipping subgroup analysis.")

    