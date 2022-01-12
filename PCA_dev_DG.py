import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

#%% Set data paths:
path_BTC_Data = '/Users/dennisgrobe/Nextcloud/Documents/Coding/Portfolio Management/Bitcoin-Fork/data_main/BTC_Data.csv'

# read data and interpolate
data = pd.read_csv(path_BTC_Data, sep=',', index_col='Date').drop(columns=['Unnamed: 0'])
data.index.name = 'Date'
data.interpolate(axis=0, inplace=True)

#%% Set interval
# interval3 = (data.index >= '2013/04/01') & (data.index <= '2021/12/31')
interval3 = (data.index >= '2013/04/01') & (data.index <= '2019/12/31')
df = data.loc[interval3]
df.head()
X = df.iloc[:, 1:]
X.head()

#%% estimators set up for scaling
estimators = []
estimators.append(['minmax', MinMaxScaler(feature_range=(-1, 1))])  # todo: look into Pipeline function!!! Sounds Cool AF
scale = Pipeline(estimators)
# Scale X (NOT = df!!!!) #
X = scale.fit_transform(X)  # todo: why do they transform that heavily? --> PCA? NOTE: They don't transform priceUSD

#%% Perform PCA
pca = PCA(n_components=6, random_state=7)  # todo: Cool Stuff!!! --> Redefine the default values of an existing function
pca.fit(X)
pca.explained_variance_ratio_  # what the code does here?
np.cumsum(pca.explained_variance_ratio_)  # todo: find out what the code does here

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Explained Variance')
plt.show()

# Create dataframe of Principal Components
df_pca = pd.DataFrame(pca.components_).transpose()

df_pca.columns = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'comp_6']

# Extract column 'priceUSD' from DF and add to df_pca
y = df.iloc[:, 0:1]
y.reset_index(drop=True, inplace=True)
df_pca['priceUSD'] = y
df_pca.head()

# write output file
path_PCA_output = '/Users/dennisgrobe/Nextcloud/Documents/Coding/Portfolio Management/Bitcoin-Fork/PCA_output/'
df_pca.to_csv(path_PCA_output + 'pca_75_reg.csv', index=False)  # todo: check if name of csv is adequate

#%% Classification

one = data['priceUSD'].shift(-1, fill_value=1)  # why fill with 1 and not just cut?
# df.drop(columns=['one'], inplace=True)  # todo: check if necessary
df['one'] = one.loc[interval3]

# create the lag of first difference of the priceUSD (CAUTION WITH THE TIME INDEX)
# inote: this will be the target variable (y[t]) in the paper
df['difference'] = ((df['one'] - df['priceUSD']) / df['priceUSD']) * 100
# df.reset_index(drop=True, inplace=True)  # why dropping index? # inote: dropped because stupid

# inote: Create dummy variable for classification if y[t] > 0
category = []
for x in range(len(df['difference'])):
    if df['difference'][x] >= 0:
        category.append(1)
    else:
        category.append(0)

sum(category)

# append classification variable to dataframe
df.insert(738, 'category', category)
df.tail()
df['priceUSD'] = df['category']  # why? inote: AHHHH, regress PCA's on categorical variable!!!

df.drop(columns=['category', 'one', 'difference'], inplace=True)

df.head(3)

X2 = df.iloc[:, 1:]
X2.head()
estimators = []
estimators.append(['minmax', MinMaxScaler(feature_range=(-1, 1))])
scale = Pipeline(estimators)

#%% PCA 2
X2 = scale.fit_transform(X2)
pca_2 = PCA(n_components=50, random_state=7)
pca_2.fit(X2)
pca_2.explained_variance_ratio_
np.cumsum(pca_2.explained_variance_ratio_)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca_2.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Explained Variance')
plt.show()

#%% PCA
df_pca_2 = pd.DataFrame(pca_2.components_).transpose()
y_2 = df.iloc[:, 0:1]
y_2.reset_index(drop=True, inplace=True)
df_pca_2['priceUSD'] = y_2
df_pca_2.head()
df_pca_2.to_csv(path_PCA_output + 'pca_75_clas.csv', index=False)
