#%%

# importing training data


import pandas as pd

iowa_file_path = 'data/home_data_for_ml_course/train.csv'

home_data = pd.read_csv(iowa_file_path)

print(home_data.describe())
home_data_describe = home_data.describe()



#%%
# Excercise 1

avg_log_size = int(home_data_describe['LotArea']['mean'].round())

newest_home_age = 2020 - home_data_describe['YearBuilt']['max']

#%% First machine learning model

# Specifing prediction target

col = home_data.columns

y = home_data.SalePrice

# Create Feature List

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_names]

#%% Print Stats

print(X.describe())
print(X.head())

#%% Fit Model

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(X, y)

#%% Make Predictions

predictions = iowa_model.predict(X)

print(predictions)
print(y)


#%% test/train split and Model validation

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(test_X)
print(mean_absolute_error(test_y, val_predictions))

#%% Finding optimal max leaf node


def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_val)
    return mae


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size, train_X, test_X, train_y, test_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print(best_tree_size)
print(scores[best_tree_size])

#%% Random Forest

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(test_X)
print(mean_absolute_error(test_y, melb_preds))