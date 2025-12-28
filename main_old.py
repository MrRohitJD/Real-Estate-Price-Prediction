import numpy as np
import pandas  as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import  Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


housing  = pd.read_csv("_06_first_ML_project\housing.csv")


housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)


split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2, random_state=42)

for train_index , test_index in split.split(housing, housing["income_cat"]):
    strat_train_set  = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set  = housing.loc[test_index].drop("income_cat", axis=1).to_csv("_06_first_ML_project/test_set.csv", index=False)
    print("saved")



housing = strat_train_set.copy()

housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value',  axis=1).copy()

num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)


feature_names = full_pipeline.get_feature_names_out()

housing_prepared_df = pd.DataFrame(
    housing_prepared,
    columns=feature_names,
    index=housing.index
)
# print(housing_prepared_df)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared_df, housing_labels)
lin_preds = lin_reg.predict(housing_prepared_df)
# lin_rmse = root_mean_squared_error(housing_labels , lin_preds)
lin_rmse = -cross_val_score(lin_reg, housing_prepared_df, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
lin_rmse_pd_ser = pd.Series(lin_rmse).describe()
print(f"LinearRegression------->{lin_rmse_pd_ser}")


dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared_df, housing_labels )
dec_preds = dec_reg.predict(housing_prepared_df)
# dec_rmse = root_mean_squared_error(housing_labels , lin_preds)
dec_rmse = -cross_val_score(dec_reg , housing_prepared_df, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
dec_rmse_pd_ser = pd.Series(dec_rmse).describe()

print(f"DecisionTreeRegressor--->{dec_rmse_pd_ser}")


forest_reg  = RandomForestRegressor(random_state=42).fit(housing_prepared_df, housing_labels)
forest_preds =forest_reg.predict(housing_prepared_df)
# forest_rmse = root_mean_squared_error(housing_labels , lin_preds)
forest_rmse = -cross_val_score(forest_reg ,housing_prepared_df, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
forest_rmse_pd_ser = pd.Series(forest_rmse).describe()
print(f"RandomForestRegressor --------->{forest_rmse_pd_ser}")



