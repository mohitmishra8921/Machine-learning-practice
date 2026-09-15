import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


 

# 1. load the dataset
housing = pd.read_csv("Exact_housing_cleaned_data.csv")   # fixed: read_csv (was reaad_csv)

# 2. create income category column for stratified sampling
housing["income_cat"] = pd.cut(                            # fixed: pd.cut, not housing.cut; column name matches usage below
    housing["Median_income"],                              # fixed: match actual column name/case in your CSV
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],                 # fixed: bins must be strictly increasing (1 -> 1.5)
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data
housing = strat_train_set.copy()

# 3. Separate predictors and labels
housing_labels = housing["Median_house_value"].copy()
housing_features = housing.drop("Median_house_value", axis=1)

# 4. Split numerical and categorical attributes
num_attribs = housing.drop("Ocean_proximity", axis=1).columns.tolist()  # fixed: use correct column name/case, get column list not a df
cat_attribs = ["Ocean_proximity"]                                       # fixed: match actual column name/case

# 5. Pipelines
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoding", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),   # fixed: matches variable name defined above
    ("cat", cat_pipeline, cat_attribs),   # fixed: matches variable name defined above
])

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

feature_names = full_pipeline.get_feature_names_out()
housing_prepared_df = pd.DataFrame(housing_prepared, columns=feature_names, index=housing.index)
print(housing_prepared_df)

# 7. Train the model

# Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = mean_squared_error(housing_labels, lin_preds)
#print(f"The root mean squared error for Linear Regression is {lin_rmse}")

lin_rmses = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                             )
print(f"The root mean squared error for Decision Tree is {lin_rmses}")

#print(pd.Series(lin_rmses).describe())

# Decision Tree Model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared, housing_labels)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = mean_squared_error(housing_labels, dec_preds)

dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                             )
print(f"The root mean squared error for Decision Tree is {dec_rmses}")
#print(pd.Series(dec_rmses).describe())

# Random Forest Model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmse = mean_squared_error(housing_labels, random_forest_preds)

random_forest_rmses = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10
                             )
print(f"The root mean squared error for Random Forest is {random_forest_rmses}")                             
# print(f"The root mean squared error for Decision Tree is {dec_rmses}")
print(pd.Series(random_forest_rmses).describe())