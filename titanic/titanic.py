# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')

def transform_df(df):
    # 性别
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

    # 头衔
    df['Title'] = df.Name.str.extract('(\w*)\.', expand=False)
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    vc = df.Title.value_counts()
    topTitles = vc.index[:4].values
    rareTitles = vc.index[4:].values
    df['Title'] = df['Title'].replace(rareTitles, 'Rare')

    i = 0
    for title in np.append(topTitles, 'Rare'):
        df.loc[(df['Title'] == title), 'Title'] = i
        i += 1

    df['Title'] = df['Title'].astype(int)

    # 利用Title补全age
    titles = df.Title.unique()
    guess_ages = {}
    for title in titles:
        guess_ages[title] = df.loc[df['Title'] == title].Age.median()
    for title in titles:
        df.loc[df.Age.isnull() & (df.Title == title), 'Age'] = guess_ages[title]

    # 独身或和家人一起
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # 补全Embarker字段
    freq_port = df.Embarked.dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 补全Fare，并且分档
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    fare_band, bins = pd.qcut(train_df['Fare'], 4, retbins=True)
    df.loc[ df['Fare'] <= bins[1], 'Fare'] = 0
    df.loc[(df['Fare'] > bins[1]) & (df['Fare'] <= bins[2]), 'Fare'] = 1
    df.loc[(df['Fare'] > bins[2]) & (df['Fare'] <= bins[3]), 'Fare']   = 2
    df.loc[ df['Fare'] > bins[3], 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # 清理字段
    df = df.drop(['Cabin', 'Ticket', 'Name', 'SibSp', 'Parch', 'PassengerId', 'FamilySize'], axis=1)

    return df

def make_prediction(clf):
    test_df = pd.read_csv('./test.csv')
    train_df = pd.read_csv('./train.csv')
    transformed_train_df = transform_df(train_df)
    transformed_test_df = transform_df(test_df)
    clf.fit(transformed_train_df.drop('Survived', axis=1), transformed_train_df['Survived'])
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': clf.predict(transformed_test_df)
    })
    submission.to_csv('submission.csv', index=False)

transformed_df = transform_df(train_df)
# 首先将数据划分成训练集和测试集，再将训练集划分成训练集和验证集
# 训练集 - 训练模型；验证集 - 调整超参数；测试机 - 选择模型
x_train, x_test, y_train, y_test = train_test_split(transformed_df.drop('Survived', axis=1), transformed_df['Survived'], test_size=0.4, random_state=0)

clfs = {
    'SVC': SVC(),
    'LinearSVC': LinearSVC(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'Perceptron': Perceptron(),
    'SGDClassifier': SGDClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}
'''
for name, clf in clfs.iteritems():
    scores = cross_val_score(clf, x_train, y_train, cv=10)
    print("%s Accuracy: %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))
'''

'''
划分训练集、验证集合、测试集的几点：
1 shuffle 打乱样本次序
2 stratified 保证集合中每类样本的比例基本一致
3 group 需要对样本分组，按组划分
4 time series
'''

test_clf = clfs['KNeighborsClassifier'].fit(x_train, y_train)
print test_clf.score(x_test, y_test)

# make_prediction(clfs['KNeighborsClassifier'])
