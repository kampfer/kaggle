# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

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

print transform_df(train_df).head(), train_df.head()

