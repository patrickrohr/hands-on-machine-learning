#!./../../env/bin/python3

# ^ not sure if this actually does anything...

## Housing Project out of Chapter 2 in Hands-On Machine Learning by Aurelien Geron.

import pandas as pd
import numpy as np

from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


##
## @brief      Class for data import.
## @date       November 27, 2017
## @author     Patrick Rohr
##
class DataImport:
    ##
    ## @brief      Constructs the object. Loads CSV and splits data into training and test sets.
    ## @param      self  The object
    ## @param      csv   Path string to data CSV file.
    ## @date       November 27, 2017
    ## @author     Patrick Rohr
    ##
    def __init__(self, csv, target_label):
        pd_data = pd.read_csv(csv)
        pd_train, pd_test = train_test_split(pd_data, test_size=0.20, random_state=42)
        self.pd_train_data, self.pd_train_labels = self._create_target(pd_train, target_label)
        self.pd_test_data, self.pd_test_labels = self._create_target(pd_test, target_label)

    def _create_target(self, data, target_label):
        result_data = data.drop(target_label, axis=1)
        result_labels = data[target_label].copy()
        return (result_data, result_labels)

##
## @brief      Class for combined attributes adder.
## @date       November 28, 2017
## @author     Patrick Rohr
##
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    ##
    ## @brief      Constructor
    ## @param      self           The object
    ## @param      rooms_ix       The rooms index
    ## @param      household_ix   The household index
    ## @param      population_ix  The population index
    ## @param      bedrooms_ix    The bedrooms index
    ## @date       November 28, 2017
    ## @author     Patrick Rohr
    ##
    def __init__(self, rooms_ix, household_ix, population_ix, bedrooms_ix):
        self.rooms_ix = rooms_ix
        self.household_ix = household_ix
        self.population_ix = population_ix
        self.bedrooms_ix = bedrooms_ix

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

# convert DataFrame to numpy.ndarray by using df.values

data = DataImport("~/Projects/HandsOnMachineLearning/datasets/housing/housing.csv", "median_house_value")


# ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity']
#print((train_data))

numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
category_features = ['ocean_proximity']
features = numeric_features + category_features

mapper = DataFrameMapper([
        (numeric_features, None),
        (category_features, LabelBinarizer())
    ])

pipeline = Pipeline([
        ('mapper', mapper),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(features.index("total_rooms"), features.index("households"), features.index("population"), features.index("total_bedrooms"))),
        ('std_scaler', StandardScaler())
    ])



data_preprocessed = pipeline.fit_transform(data.pd_train_data)

print(data_preprocessed.shape)
#print(result)

# linear_regression = LinearRegression()
# linear_regression.fit(data_preprocessed, data.pd_train_labels)

# some_data = data.pd_train_data.iloc[10:15]
# some_labels = data.pd_train_labels.iloc[10:15]
# some_data_prepared = pipeline.transform(some_data)
# print("Predictions", linear_regression.predict(some_data_prepared))
# print("Labels", list(some_labels))

# housing_predictions = linear_regression.predict(data_preprocessed)
# lin_mse = mean_squared_error(data.pd_train_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# print("-----------")

# dt_reg = DecisionTreeRegressor()
# dt_reg.fit(data_preprocessed, data.pd_train_labels)

# housing_predictions = dt_reg.predict(data_preprocessed)
# dt_rmse= np.sqrt(mean_squared_error(data.pd_train_labels, housing_predictions))
# print(dt_rmse)

# Cross Validation

def display_scores(scores):
    scores = np.sqrt(-scores)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

scores = cross_val_score(DecisionTreeRegressor(), data_preprocessed, data.pd_train_labels, scoring="neg_mean_squared_error", cv=10)
print("DecisionTreeRegressor")
display_scores(scores)

scores = cross_val_score(LinearRegression(), data_preprocessed, data.pd_train_labels, scoring="neg_mean_squared_error", cv=10)
print("LinearRegressor")
display_scores(scores)


scores = cross_val_score(RandomForestRegressor(), data_preprocessed, data.pd_train_labels, scoring="neg_mean_squared_error", cv=10)
print("LinearRegressor")
display_scores(scores)