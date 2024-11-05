import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder


################################
# Principal Component Analysis
################################

df = pd.read_csv("_BOOTCAMP_/datasets/hitters.csv")
df.head()

#bizi bu problem için sadece numerik değişkenler ilgilendiriyor. Target değişken olan Salary değişkeni ile de ilgilenmiyoruz.
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace=True)
df.shape


# amacımız 16 değişkenli bu veri setini 2-3 değişkene indirgemek
df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

################################
# Final PCA'in Oluşturulması
################################

# 3 bileşen seçelim

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
# array([0.46037855, 0.26039849, 0.1033886 ]) (sırasıyla değişkenlerin tek başına açıklama oranı)

np.cumsum(pca.explained_variance_ratio_)
# array([0.46037855, 0.72077704, 0.82416565])  (kümülatif açıklama oranı)

################################
# Principal Component Regression
################################

df = pd.read_csv("_BOOTCAMP_/datasets/hitters.csv")
df.shape  # (322, 20)

len(pca_fit)  # 322

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols) # 16

others = [col for col in df.columns if col not in num_cols]  # num_cols un dışında kalan değişkenler

pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

#      PC1    PC2    PC3
# 0 -3.240 -0.253  0.776
# 1  0.245  1.303  0.118
# 2  0.604 -2.617 -0.698
# 3  3.591  0.548 -1.049
# 4 -2.265 -0.699 -1.291

# Bu case için bir problem varsaydık: Çoklu doğrusal bağlantı problemi. Yani değişkenlerin birbiri ile yüksek korelasyonlu olması problemi.
# Ve indirgeme yaparak bu problemi ortadan kaldırdık. Ortaya çıkan bu 3 değişkenin birbiri ile korelasyonuNU ortadan kaldırdık.

df[others].head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),   #çıkardığımız değişkenleri tekrar ekliyoruz.
                      df[others]], axis=1)
final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
# 345.6021106351967

y.mean()
# 535.9258821292775

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
# 373.27960849731846

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV hiperparametre optimizasyonu
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
# 330.1964109339104

################################
# PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("_BOOTCAMP_/datasets/breast_cancer.csv")


# bu çok değişkenli veriyi iki eksen üzerinde görselleştirmek istiyoruz

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)


def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)  # PCA için X in her zaman sayısal değişkenler olması lazım

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("_BOOTCAMP_/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")




















