"""
Data Can be found at
https://www.kaggle.com/c/titanic/data

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# loading data

train_data = pd.read_csv('data/titanic/train.csv')
test_data = pd.read_csv('data/titanic/test.csv')



# rate that a gender has survived

# women

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_woman = sum(women)/len(women)

print(f"{rate_woman} of women survived")

# men

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)

print(f"{rate_men} of men survived")

# Random forest predictor

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('titanic_challange/data/my_submission.csv', index=False)

