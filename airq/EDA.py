# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np, os
from scipy.stats import pearsonr, spearmanr

# Load the dataset
df = pd.read_excel(r'C:\Users\pc\Downloads\data.xlsx', parse_dates= True)
df.set_index('datetime', inplace=True)
# Check the first few rows of the dataset
print(df.head())

# Check the basic information about the dataset
print(df.info())

# Check the statistical summary of the dataset
print(df.describe())

# Check for missing values in the dataset
print(df.isnull().sum())

# Drop the data after 2019

# Visualize the distribution of each variable using histograms
df.hist(bins=50, figsize=(20,15))
plt.show()

#mask = np.zeros_like(df.corr(), dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

# Visualize the correlations between variables using a heatmap
sns.heatmap(df.corr(), cmap='coolwarm', vmin=-0.9, vmax=0.9, annot=True, fmt='.2f')
plt.show()

# Visualize the relationship between the target variable and the other variables using scatterplots
sns.pairplot(df, x_vars=['ws', 'wd', 'temp', 'dew_temp', 'pressure', 'wv', 'blh', 'bcaod550', 'duaod550', 'omaod550', 'ssaod550', 'suaod550', 'aod469', 'aod550', 'aod670', 'aod865', 'aod1240'], y_vars=['pm2p5'], height=7, aspect=0.7)
plt.show()


# , height=8, aspect=0.5


from sklearn.feature_selection import mutual_info_regression

X = df.copy()
y = X.pop("pm2p5")

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y)
mi_scores[::3]  # show a few features with their MI scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


# Calculate Pearson correlation coefficient
corr_p, p_val_p = pearsonr(df['pm2p5'], df['pressure'])
print("Pearson correlation coefficient:", corr_p)
print("p-value:", p_val_p)

# Calculate Spearman correlation coefficient
corr_s, p_val_s = spearmanr(df['pm2p5'], df['pressure'])
print("Spearman correlation coefficient:", corr_s)
print("p-value:", p_val_s)








