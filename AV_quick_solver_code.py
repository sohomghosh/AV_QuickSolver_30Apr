import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

train=pd.read_csv("/home/sohom/Desktop/AV_quick_solver/train.csv")
test=pd.read_csv("/home/sohom/Desktop/AV_quick_solver/test.csv")
user=pd.read_csv("/home/sohom/Desktop/AV_quick_solver/user.csv")
article=pd.read_csv("/home/sohom/Desktop/AV_quick_solver/article.csv")

set(list(test['ID']))-set(list(train['ID']))
#No extra items in test set

train_user=train.merge(user, left_on='User_ID', right_on='User_ID', how='left')
train_user_article=train_user.merge(article,left_on='Article_ID', right_on='Article_ID', how='left')
train_user_article.to_csv("/home/sohom/Desktop/AV_quick_solver/train_user_article.csv",index=False)

user_ids_list=[int(str(id).split('_')[0]) for id in list(test['ID'])]
article_ids_list=[int(str(id).split('_')[1]) for id in list(test['ID'])]
test=test.assign(User_ID=user_ids_list)
test=test.assign(Article_ID=article_ids_list)
test_user=test.merge(user, left_on='User_ID', right_on='User_ID', how='left')
test_user_article=test_user.merge(article,left_on='Article_ID', right_on='Article_ID', how='left')
test_user_article.to_csv("/home/sohom/Desktop/AV_quick_solver/test_user_article.csv",index=False)

test_user_article['Rating']=float('nan')
train_test = train_user_article.append(test_user_article)

train_user_article['Age'].value_counts()
age_encode = {'Less tha 15': 0, '15-20': 1, '20-30': 2, '30-40': 3, '40-50': 4,'50-60':5,'60-70':6,'More than 70':7}
train_test['Age'] = train_test['Age'].replace(to_replace = age_encode)

train_user_article['Var1'].value_counts()
lbl = LabelEncoder()
lbl.fit(list(train_test['Var1'].values))
train_test['Var1'] = lbl.transform(list(train_test['Var1'].values))

features = np.setdiff1d(train_test.columns, ['ID', 'User_ID', 'Article_ID','Rating'])

X_train_valid=train_test[0:len(train.index)]
X_train=X_train_valid.sample(frac=0.1, replace=False)
X_valid=pd.concat([X_train_valid, X_train]).drop_duplicates(keep=False)
X_test=train_test[len(train.index):len(train_test.index)]

dtrain = xgb.DMatrix(X_train[features].as_matrix(columns=None), X_train['Rating'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features].as_matrix(columns=None), missing=np.nan)
dtest = xgb.DMatrix(X_test[features].as_matrix(columns=None), missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]


params = {"objective": "reg:linear","booster": "gblinear", "nthread": 4, "silent": 1,
                "eta": 0.1, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.8,
                "min_child_weight": 1, "eval_metric":"rmse","gamma":0,
                "seed": 0, "scale_pos_weight":1,"tree_method": "auto"}

'''
'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]

 'min_child_weight':[6,8,10,12]

 'gamma':[i/10.0 for i in range(0,5)]

 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]

 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]

 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]

 learning_rate =0.01,

'''

params = {"objective": "reg:linear","booster": "gblinear", "nthread": 4, "silent": 0,
                "eta": 0.1, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.25,"subsample":0.8,
                "min_child_weight": 2, "eval_metric":"rmse","gamma":0,
                "seed": 0, "scale_pos_weight":1,"tree_method": "auto"}

params = {"objective": "reg:linear","booster": "gblinear", "nthread": 4, "silent": 0,
                 "eta": 0.15, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.2,"subsample":0.8,
                 "min_child_weight": 2, "eval_metric":"rmse","gamma":0,
                 "seed": 0, "scale_pos_weight":1,"tree_method": "auto","reg_alpha":0}
 


bst = xgb.train(params, dtrain)
valid_preds= bst.predict(dvalid)
print(mean_squared_error(list(X_valid['Rating']), valid_preds))

test_preds = bst.predict(dtest)
test_preds=[max(0,i) for i in test_preds]
submit = pd.DataFrame({'ID': test['ID'], 'Rating': test_preds})
submit.to_csv("xgb_7.csv", index=False)





##############################Experimenting with h2o#############################################################################
import h2o

# initialize an h2o cluster
h2o.init()
#h2o.init(nthreads = -1,max_mem_size = "6G") 
h2o.connect()
train = h2o.import_file("/home/sohom/Desktop/AV_quick_solver/train_user_article.csv")
test = h2o.import_file("/home/sohom/Desktop/AV_quick_solver/test_user_article.csv")

train["Var1"]=train["Var1"].asfactor()
test["Var1"]=test["Var1"].asfactor()

train["Rating"]=train["Rating"].asfactor()


r = train.runif()   
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features_h2o = list(np.setdiff1d(train.columns, ['ID', 'User_ID', 'Article_ID','Rating']))

#GBM
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features_h2o, y="Rating", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred= pred[:,1]
submit_pred.head()
submission_dataframe =(test[:,'ID']).cbind(submit_pred)
submission_dataframe.set_name(1,"Ratings")
h2o.h2o.export_file(submission_dataframe, path ="submission_gbm_1.csv")



pred = model_gbm.predict(test)
pred.head()
submit_pred= pred[:,1]
submit_pred.head()
submission_dataframe =(test[:,'ID']).cbind(submit_pred)
submission_dataframe.set_name(1,"Ratings")
h2o.h2o.export_file(submission_dataframe, path ="submission_gbm_1.csv")
