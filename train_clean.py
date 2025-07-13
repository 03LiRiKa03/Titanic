import pandas as pd

train = pd.read_csv("train.csv")

median_age = train['Age'].median()
train['Age'] = train['Age'].fillna(median_age)

train['HasCabin'] = train['Cabin'].notna().astype(int)
train.drop(columns = ['Cabin'], inplace = True)

most_common_embarked = train['Embarked'].mode()[0]
train['Embarked'].fillna(most_common_embarked, inplace = True)

 



