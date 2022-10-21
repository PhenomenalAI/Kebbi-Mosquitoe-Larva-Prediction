# Import Libaries

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr
from scipy import stats

import math

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as ltb

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

import warnings


# Training



# Load dataset
data = pd.read_csv("wetSSN.csv")

# Any correlation?
data.corr()

# Drop unecessary columns
target = "No_of_Larv"
y = data[target]
X = data.drop(columns = ["FID", "Eastings", "Northings", target])

# Split dataset to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 7)

# Compute baaseline
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

# Random Forest Model
ForestModel = RandomForestRegressor(criterion = "absolute_error", n_estimators = 250, min_samples_split = 8, min_samples_leaf = 2)
ForestModel.fit(X_train, y_train)

y_pred = ForestModel.predict(X_test)

# Gradient Boosting
GB = GradientBoostingRegressor(n_estimators = 300, learning_rate = 1.0,max_depth = 1, random_state = 0).fit(X_train, y_train)
GB.fit(X_train, y_train)

y_pred = GB.predict(X_test)

# Extreme Gradient Boosting
xg_reg = XGBRegressor()
xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)

# CatBoosting
model = CatBoostRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# LightBoosting
ltbmodel = ltb.LGBMRegressor()
ltbmodel.fit(X_train, y_train)

y_pred = ltbmodel.predict(X_test)



# Prediction with best model


# Load dataset
test = pd.read_csv("drySSN.csv")

# Saving Coordinate
# Dropping unecessary columns
Easting = test.Eastings
Northing = test.Northings

test = test.drop(['Eastings', 'Northings', 'FID'], axis = 1)

NoL = ForestModel.predict(test)

stats.describe(NoL)

# Adding back the coordinate to the predicted value
Easting = Easting.values
Northing = Northing.values

txt = {'Eastings' : Easting, 'Northings' : Northing, 'No_of_Larv' : NoL}
txt = pd.DataFrame(txt)
txt.describe()

# Saving to csv
txt.to_csv("Kebbi_DSS_Pred.csv")