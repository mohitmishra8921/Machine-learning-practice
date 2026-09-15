import os 
import joblib
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

Model_file  = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"   # NOT: Pipeline = "pipeline.pkl"



def build_pipeline(num_attribs,cat_attribs):
        num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler())])

        cat_pipeline = Pipeline([
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Full pipeline
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),   # fixed: matches variable name defined above
            ("cat", cat_pipeline, cat_attribs),   # fixed: matches variable name defined above
        ])

        return full_pipeline
if not os.path.exists(Model_file):
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
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv",index = False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)
        
        housing_labels = housing["Median_house_value"].copy()
        housing_features = housing.drop("Median_house_value", axis=1)

        num_attribs = housing_features.drop("Ocean_proximity", axis=1).columns.tolist()  # fixed: use correct column name/case, get column list not a df
        cat_attribs = ["Ocean_proximity"]   


    pipeline = build_pipeline(num_attribs,cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    print( housing_prepared )

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    joblib.dump(model,Model_file)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained .Congrats!")

else:
    model = joblib.load(Model_file)    
    pipeline= joblib.load(PIPELINE_FILE)    

    input_data = pd.read_csv("input.csv")

    transform_input = pipeline.transform(input_data)
    predictions = model.predict(transform_input)
    input_data["Median_house_value"] = predictions

    input_data.to_csv("output.csv",index = False)
    print("Inference is complete result is saved as output.csv Enjoy!")





