# coding: utf-8

import pandas as pd

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

import gc
import pickle

data = pd.read_pickle('data.pkl')

data = data[[
	'date_block_num',
	'shop_id',
	'item_id',
	'item_cnt_month',
	'city_code',
	'item_category_id',
	'type_code',
	'subtype_code',
	'item_cnt_month_lag_1',
	'item_cnt_month_lag_2',
	'item_cnt_month_lag_3',
	'item_cnt_month_lag_6',
	'item_cnt_month_lag_12',
	'date_avg_item_cnt_lag_1',
	'date_item_avg_item_cnt_lag_1',
	'date_item_avg_item_cnt_lag_2',
	'date_item_avg_item_cnt_lag_3',
	'date_item_avg_item_cnt_lag_6',
	'date_item_avg_item_cnt_lag_12',
	'date_shop_avg_item_cnt_lag_1',
	'date_shop_avg_item_cnt_lag_2',
	'date_shop_avg_item_cnt_lag_3',
	'date_shop_avg_item_cnt_lag_6',
	'date_shop_avg_item_cnt_lag_12',
	'date_cat_avg_item_cnt_lag_1',
	'date_shop_cat_avg_item_cnt_lag_1',
	#'date_shop_type_avg_item_cnt_lag_1',
	#'date_shop_subtype_avg_item_cnt_lag_1',
	'date_city_avg_item_cnt_lag_1',
	'date_item_city_avg_item_cnt_lag_1',
	#'date_type_avg_item_cnt_lag_1',
	#'date_subtype_avg_item_cnt_lag_1',
	'delta_price_lag',
	'month',
	'days',
	'item_shop_last_sale',
	'item_last_sale',
	'item_shop_first_sale',
	'item_first_sale',
]]

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect();

# defining estimate

rf1 = RandomForestRegressor(n_estimators=75, min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=11, n_jobs=4)
rf2 = RandomForestRegressor(n_estimators=88, min_samples_leaf=25, max_features='sqrt', oob_score=True, random_state=42, n_jobs=8)
rf3 = RandomForestRegressor(n_estimators=57, min_samples_leaf= 1, max_features='sqrt' ,oob_score=True, random_state=39, n_jobs=8)
rf5 = RandomForestRegressor(n_estimators=75, min_samples_leaf=14, max_features='sqrt' ,oob_score=True, random_state=11, n_jobs=4)

# rf4 comes from a GridSearchCV method
param_grid = {'n_estimators': list(range(35, 100, 20)), 'min_samples_leaf': list(range(1, 25, 10))}
rfr = RandomForestRegressor(max_features='sqrt', oob_score=True, random_state=11, n_jobs=4)
rf4 = GridSearchCV(rfr, param_grid = param_grid, n_jobs=4, verbose=1)

print("Start fitting...")

rf1.fit(X_train,Y_train)
rf2.fit(X_train,Y_train)
rf3.fit(X_train,Y_train)
rf4.fit(X_train,Y_train)
rf5.fit(X_train,Y_train)

print("Start predicting...")

test1 = rf1.predict(X_test).clip(0, 20)
test2 = rf2.predict(X_test).clip(0, 20)
test3 = rf3.predict(X_test).clip(0, 20)
test4 = rf4.predict(X_test).clip(0, 20)
test5 = rf5.predict(X_test).clip(0, 20)

Y_test = (test1+test2+test3+test4+test5)/5

# for index generation
test = pd.read_csv('test.csv').set_index('ID')

submission = pd.DataFrame({
	"ID": test.index, 
	"item_cnt_month": Y_test
})
submission.to_csv('rf_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_test, open('rf_test.pickle', 'wb'))
