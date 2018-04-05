# Vladimir Denisov's stacker modified to return averages for different ensemble parameters

# This version outputs the OOF predictions to use later 
# for validating the stacker and meta-ensembling

import sys
sys.path.append('/usr/local/lib/python3.4/site-packages')

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, roc_curve, auc

import lightgbm as lgb
from lightgbm import LGBMClassifier

train = pd.read_csv('D:/PYTHON FILES/Claim data/train.csv')
test = pd.read_csv('D:/PYTHON FILES/Claim data/test.csv')


# Preprocessing 
id_test = test['id'].values
print(id_test)
target_train = train['target'].values
id_train = train['id'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

#train = train.replace(-1, np.nan)
#test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)


print(train.values.shape, test.values.shape)

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['objective'] = 'binary'
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['lambda_l2'] = 0.3
lgb_params['min_data'] = 1
lgb_params['num_leaves'] = 15
lgb_params['max_depth'] = 5
lgb_params['verbose']=500

trainarr = np.asarray(train)
target_trainarr = np.asarray(target_train)
testarr = np.asarray(test)

def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) -1
    return g
    
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True

lgb_model = LGBMClassifier(**lgb_params)
base_params = [lgb_params]

sfs = RFE(estimator=lgb_model, n_features_to_select=55, step=0.2)

folds = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=2016).split(trainarr, target_trainarr))
        
S_pred_full = np.zeros((testarr.shape[0]))
        
for i, params in enumerate(base_params, start=1):
        
    for j, (train_idx, test_idx) in enumerate(folds, start=1):
        
        print ("Train %d fold %d" % (i, j))
                
        X_train = trainarr[train_idx]
        y_train = target_trainarr[train_idx]
        X_holdout = trainarr[test_idx]
        y_holdout = target_trainarr[test_idx]
        
        print('calling feature selection')
        sfs1 = sfs.fit(X_train, y_train)
        print('transforming')
        X_train_sfs = sfs1.transform(X_train)
        X_holdout_sfs = sfs1.transform(X_holdout)
        X_test_sfs = sfs1.transform(testarr)
        
        print(X_train_sfs.shape)
        print(y_train.shape)
        clf_train = lgb.Dataset(X_train_sfs, label=y_train, free_raw_data=False)
        #lgb_model.fit(X_train_sfs,y_train)
        clf_valid = lgb.Dataset(X_holdout_sfs, label=y_holdout, reference=clf_train, free_raw_data=False)
        print('calling model')
        print(X_holdout_sfs.shape)
        print(y_holdout.shape)
        #print(cross_val_score(lgb_model, X_holdout_sfs, y_holdout))
        model = lgb.train(params, clf_train, 2500, clf_valid, verbose_eval=50, feval=gini_lgb, early_stopping_rounds=200)
        S_pred_full += model.predict(X_test_sfs, num_iteration=model.best_iteration)
        #S_pred_full += lgb_model.predict_proba(X_test_sfs)[:,1]
        
    print('pred model')
    S_pred_M = S_pred_full/j    
        
print('pred final')
S_pred_F = S_pred_full/(i*j)
        
stack_res = S_pred_F

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = stack_res
sub.to_csv('D:/PYTHON FILES/Claim data/stacked_avg_lgb.csv', index=False)