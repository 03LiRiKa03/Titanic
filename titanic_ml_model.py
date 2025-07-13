import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train.csv")

median_age = train['Age'].median()
most_common_embarked = train['Embarked'].mode()[0]

train['Age'].fillna(median_age, inplace=True)
train['HasCabin'] = train['Cabin'].notna().astype(int)

train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
train['Embarked'].fillna(most_common_embarked, inplace=True)
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

test = pd.read_csv("test.csv")

passenger_ids = test['PassengerId'].copy()

test['Age'].fillna(median_age, inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['HasCabin'] = test['Cabin'].notna().astype(int)

test['Embarked'].fillna(most_common_embarked, inplace=True)

test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin']

X_train = train[features]
y_train = train['Survived']
X_test  = test[features]


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)