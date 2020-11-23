#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:52:29 2020

@author: albwrekt
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats

# This is the absolute path of the datasets
HOUSING_PATH = "~/oreilly_ml/ml/handson-ml2/datasets/housing"

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

# custom transformer to add extrapolated attributes to the data
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix]/X[:, households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        

# This method loads the data into a Pandas Dataframe
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Split dataset into training and test sets to avoid data snooping bias and generalization error.
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

# This method uses CRC32 hashes to verify the same data is used for test and training set
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

# This method uses the test_set_check to set up the test and training_set based on the respective hash
def split_train_test_by_id(data, test_ratio, id_col):
    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# this method is made to display the scores of the testing
def display_scores(regression, scores):
    print("\nRegression Type: ", type(regression))
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard Deviation: ", scores.std())
    print("\n\n\n")

# Load the dataframe
housing = load_housing_data()

#This method displays the first 5 rows aka the head
print(housing.head())

# This provides the columns and datatypes
print(housing.info())

# This shows the values stored in the attribute ocean_proximity
print(housing["ocean_proximity"].value_counts())

# This method shows a statistical summary of each attribute
print(housing.describe())

# Plotting the dataset as a histogram, picture listed in this file
housing.hist(bins=50, figsize=(20,15))
#plt.show()

# Add row index as unique identifier
housing_with_id = housing.reset_index()

# Add unique identifier based on a unique data attribute to save data size
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

# Separate the training and testing sets based on the test ratio and hash calculation
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# This is the same as the method provided by scikit-learn called train_test_split

# Create median income categories to help identify strata in the data
housing_with_id["income_cat"] = pd.cut(housing_with_id["median_income"],
                                       bins=[0, 1.5, 3, 4.5, 6, np.inf],
                                       labels = [1,2,3,4,5])

# display the income categories as a graph
housing_with_id["income_cat"].hist()
#plt.show()

#Use StratifiedShuffleSplit to split the test and training_set based on the proportion of the dataset_strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_with_id, housing_with_id["income_cat"]):
    strat_train_set = housing_with_id.loc[train_index]
    strat_test_set = housing_with_id.loc[test_index]

# verify that the test set and the training_set both have equal proportionality of strata in their dataset
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

# remove the categories from the dataset
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace = True)
    set_.drop("id", axis=1, inplace=True)
    
# make copy of training_set to make sure original data is not altered
housing = strat_train_set.copy()

# visualize latitude and longitude with scatterplot
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# plot the housing prices as a scale of high to low based on color map
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population",
             figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

# computing the correlation matrix for all attributes
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#visualize the correlation data using the pandas library's scatter_matrix method
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# isolating the median income attribute due to its strong correlation to median house value
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

# create additional attributes that will be helpful to analyze the data
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# reanalyze the correlation matrix to analyze the new data values
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# reset the dataset to be fed into ML algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# accomodate missing values in the total_bedrooms field using DataFrame methods
# housing.dropna(subset=["total_bedrooms"]) # Option 1
# housing.drop("total_bedrooms", axis=1) # Option 2
# median = housing["total_bedrooms"].median() # Option 3
# housing["total_bedrooms"].fillna(median, inplace=True)

# Can use SKLearn's Simple Inputer to inject median into empty datapoints
imputer = SimpleImputer(strategy="median")

# drop the text based fields in order to get medians
housing_num = housing.drop("ocean_proximity", axis=1)

# fit the imputer to calculate the strategy values
imputer.fit(housing_num)

# validate imputer is getting means of numeric values
print(imputer.statistics_)
print(housing_num.median().values)

# use imputer to transform the training set.
X = imputer.transform(housing_num)

# Place the data back into a Pandas dataframe object
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = housing_num.index)

# Analyze the text attribute, ocean_proximity
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

# convert the text values to numbers through an encoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

# Use One Hot Encoding to determine where each category, stored in SciPy sparse matrix to save space
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# utilizing custom transformer to modify the data
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# setup a pipeline through sklearn
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


#set up column transformer to fit, transform, and scale all columns regardless of values
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
    ])
housing_prepared = full_pipeline.fit_transform(housing)

# Run a regression on the transformed data
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#try out the regression model on the data from the test set
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

# find the RMSE of the predictions from the entire dataset
housing_linreg_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_linreg_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Linear Regression RMSE: " + str(lin_rmse))

# Try a Decision Tree Regression model
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_tree_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_tree_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision Tree Regression RMSE: " + str(tree_rmse))

# Using a Random Forest Regression
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_forest_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_forest_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest Regression RMSE: " + str(forest_rmse))

# Performing cross-validation using K-Fold through SciKit through Decision Tree Regression
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_reg, tree_rmse_scores)

# performing cross-validation using K-Fold through Scikit on Linear Regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_reg, lin_rmse_scores)

# performing cross-validation using K-Fold through SciKit on Random Forest Regression
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_reg, forest_rmse_scores)

# Using GridSearchCV to test hyper parameters on the Random Forest Regressor
param_grid = [
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 90, 270, 810], 'max_features':[4, 10, 50, 100, 1000]}]

# Including Grid Search CV results from presented parameter grids
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, refit=True)
grid_search.fit(housing_prepared, housing_labels)

# Print out the best results and analyze the best model
print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("Test Set:", np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print("Grid Search Feature Importances:", feature_importances)

# Display the feature importances next to their respective features
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

# Run the final test set on the now trained model
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# perform statistic on the final model
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
result = np.sqrt(stats.t.interval(confidence, 
                         len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print("Result:", result)

