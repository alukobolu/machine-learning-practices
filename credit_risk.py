import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy.stats import uniform
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, recall_score, make_scorer, plot_confusion_matrix, confusion_matrix, accuracy_score,f1_score

df = pd.read_csv('original.csv')
df.info()

dataframe = df
feature = 'age'
sns.set_style('ticks')
plt.figure(figsize=(10,7))
dataframe[feature].hist(bins=50)
plt.title(f"Distribution of the feature `{feature}`",fontsize=25)
plt.show()

neg_age = df[df['age'] < 0].index
df.drop(neg_age,axis=0,inplace=True)
df['age'].fillna(df['age'].mean(),inplace=True)


dataframe = df
feature1 = 'default'
sns.countplot(dataframe[feature1])
plt.title(f"Distribution of the feature `{feature1}`")

dataframe = df
feature_1 = 'default'
feature_2 = 'age'
plt.figure(figsize=(7,7))
sns.boxplot(x=feature_1, y=feature_2, data=dataframe)
plt.title("How does one's age affect risk of default?")
plt.show()

df['income'].describe()

dataframe = df
feature = 'income'
sns.set_style('ticks')
plt.figure(figsize=(10,7))
dataframe[feature].hist()
plt.title(f"Distribution of the feature `{feature}`",fontsize=25)
plt.show()

dataframe = df
feature_1 = 'default'
feature_2 = 'income'
plt.figure(figsize=(7,7))
sns.boxplot(x=feature_1, y=feature_2, data=dataframe)
plt.title("How does one's income affect risk of default?")
plt.show()

dataframe = df
cat_feat = 'default'
cont_feat = 'income'
plt.figure(figsize=(7,7))
for value in df[cat_feat].unique():
    sns.distplot(df[df[cat_feat] == value][cont_feat], label=value)
plt.legend()
plt.title(f"Distribution of `{cont_feat}` conditional on `{cat_feat}`")
plt.show()

df['loan'].describe()

dataframe = df
feature = 'loan'
sns.set_style('ticks')
plt.figure(figsize=(10,7))
dataframe[feature].hist()
plt.title(f"Distribution of {feature}",fontsize=25)
plt.show()

df[df['loan'] < 10]

dataframe = df
feature_1 = 'default'
feature_2 = 'loan'
plt.figure(figsize=(7,7))
sns.boxplot(x=feature_1, y=feature_2, data=dataframe)
plt.title("How does loan amount affect risk of default?")
plt.show()

dataframe = df
cat_feat = 'default'
cont_feat = 'loan'
plt.figure(figsize=(7,7))
for value in df[cat_feat].unique():
    sns.distplot(df[df[cat_feat] == value][cont_feat], label=value)
plt.legend()
plt.title(f"Distribution of `{cont_feat}` conditional on `{cat_feat}`")
plt.show()

dataframe = df
feature1 = 'loan'
feature2 = 'income'


g=sns.jointplot(x=dataframe[feature1], y=dataframe[feature2], kind="kde")
g.fig.set_figwidth(11)
g.fig.set_figheight(13)
plt.show()

df1 = df[['income','loan']].copy()
df1['Binned income'] = pd.cut(df1['income'],7)
df1 = df1.groupby('Binned income').std()
df1.reset_index(inplace=True)
df1['Binned income'] = df1['Binned income'].astype(str)
df1.sort_values(by=['loan'],ascending=True,inplace=False)


fig = px.bar(df1,
             x='Binned income',
             y='loan',
             title='Spread (stdev) of the loan amount for each income category')
fig.show()

X = df.drop(['clientid','default'],axis=1)
y = df['default']



sns.set_style('darkgrid')

forest_clf = RandomForestClassifier(n_estimators=100)
forest_clf.fit(X, y)

importances = forest_clf.feature_importances_
indices = np.argsort(importances)[::-1]

sns.set_style('darkgrid')

forest_clf = RandomForestClassifier(n_estimators=100)
forest_clf.fit(X, y)

importances = forest_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(7,7))
plt.bar(range(len(indices)),importances[indices])
plt.xticks(range(len(indices)), indices)
plt.title("Feature importance (Random Forest)")
plt.xlabel('Index of a feature')
plt.ylabel('Feature importance')
plt.show()



lowest_importance = X[X.columns[indices[-1]]].name
print(f'RF esimations show that the feature with the lowest importance is: {lowest_importance}')

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=15)


mmsc = MinMaxScaler()
X_train = mmsc.fit_transform(X_train)
X_test = mmsc.transform(X_test)



nb_clf = GaussianNB().fit(X_train,y_train)
print(classification_report(y_true=y_test, y_pred=nb_clf.predict(X_test)))
plot_confusion_matrix(nb_clf, X_test, y_test)



log_random_state = None
log_clf = LogisticRegression(random_state=log_random_state).fit(X_train, y_train)
print(classification_report(y_true=y_test, y_pred=log_clf.predict(X_test)))
plot_confusion_matrix(log_clf, X_test, y_test)



recall_1 = make_scorer(recall_score,pos_label=1)

MIN = 1 #Min number of neighbors
MAX = 30 #Max number of neighbors
knn_estimator = KNeighborsClassifier()
knn_clf = GridSearchCV(knn_estimator,
                       {'n_neighbors': range(MIN,MAX+1)}
                       ,scoring=recall_1).fit(X_train, y_train)  
print(knn_clf.predict(X_test))
print(classification_report(y_true=y_test, y_pred=knn_clf.predict(X_test)))
plot_confusion_matrix(knn_clf, X_test, y_test)



estimator = RandomForestClassifier(random_state=13)
rf_clf = GridSearchCV(estimator,
                      param_grid={'n_estimators':[10,20,50,100], 'criterion': ['entropy','gini']},
                      scoring=recall_1).fit(X_train, y_train)

print(rf_clf.predict(X_test))
print(classification_report(y_true=y_test, y_pred=rf_clf.predict(X_test)))
plot_confusion_matrix(rf_clf, X_test, y_test)
