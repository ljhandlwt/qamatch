# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:09:56 2017

@author: kiey
"""
#####交叉检验结果

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold

import collections

#%%读入数据，训练集
train_df = pd.read_csv("train_feature.csv")
temp = train_df.columns.tolist()
temp.sort()
train_df = train_df[temp]
#%%读入数据，测试集
test_df = pd.read_csv("test_feature.csv")
temp = test_df.columns.tolist()
temp.sort()
test_df = test_df[temp]
#%%读入数据dev集
train_dev = pd.read_csv("dev_feature.csv")
temp = train_dev.columns.tolist()
temp.sort()
train_dev = train_dev[temp]
#%%读入数据，dev.txt
def load_file(file):
	f = open(file)
	data = []
	for line in f:
		line = line.strip().split('\t')
		data.append(line)
		data[-1][0] = int(data[-1][0])
	df = pd.DataFrame(data)
	df.columns = ['label','question','answer']	

	return df

df = load_file('dev.txt')
#%%选特征
'''
train_df = train_df[['iawf1','iawf2','iawf3','label']]
train_dev = train_dev[['iawf1','iawf2','iawf3','label']]
train_df = train_df[['lcs','lcs_p','overlap','overlap_p','dist_bag_cos','dist_bag_jac','dist_bag_man','dist_vec_cos','dist_vec_man','dist_vec_euc','label']] 
train_dev = train_dev[['lcs','lcs_p','overlap','overlap_p','dist_bag_cos','dist_bag_jac','dist_bag_man','dist_vec_cos','dist_vec_man','dist_vec_euc','label']]

train_df = train_df[['lcs','lcs_p','overlap','overlap_p','dist_bag_cos','dist_bag_jac','dist_bag_man','dist_vec_cos','dist_vec_man','dist_vec_euc','iawf1','iawf2','iawf3','label']]
train_dev = train_dev[['lcs','lcs_p','overlap','overlap_p','dist_bag_cos','dist_bag_jac','dist_bag_man','dist_vec_cos','dist_vec_man','dist_vec_euc','iawf1','iawf2','iawf3','label']]
'''
#%%定义xgboost
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=321, num_rounds=10000):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.06
    param['max_depth'] = 7
    param['silent'] = 1

#    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 80
    param['subsample'] = 1
    param['max_delta_step'] = 50 
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
        pred_test_y = model.predict(xgtest,ntree_limit=model.best_iteration)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y = model.predict(xgtest)

    #pred_test_y = model.predict(xgtest)
    return pred_test_y, model

#%%拆分label
train_y = np.array(train_df["label"])
train_X = np.array(train_df.drop("label",1))

#%%使用全部train+dev
'''
train_all = pd.concat([train_df,train_dev])

temp = train_all.columns.tolist()
temp.sort()
train_all = train_all[temp]
test_df = test_df[temp]

train_y = np.array(train_all["label"])
train_X = np.array(train_all.drop("label",1))
'''
#%%交叉检验，确定num_round。
'''
best_round=[]
best_scores = []

kf=StratifiedKFold(train_y, n_folds=5,shuffle=True, random_state=15)
for dev_index, val_index in kf:
        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
        dev_y, val_y = train_y[dev_index], train_y[val_index]
        preds, model = runXGB(dev_X, dev_y, val_X, val_y)
        best_round.append(model.best_iteration)
        best_scores.append(model.best_score)

print(best_round)
print(best_scores)
'''
#%%预测dev
test_X = np.array(train_dev.drop("label",1))
preds = runXGB(train_X, train_y, test_X,num_rounds=300)

#MRR评估
df_o = pd.DataFrame(preds[0],columns=['ans'])
df_o.columns = ['ans']
df_o['label'] = df['label']

mrr = 0.0
cnt_q = 0
cnt_q_have_ans = 0

i = 0
j = 0
n = len(df_o)

list_cnt = []

while i<n:
	ans_ix = -1
	j = i
	while j<n:
		if df.at[i,'question']!=df.at[j,'question']:
			break
		if df.at[j,'label'] == 1:
			ans_ix = j
		j += 1
	if ans_ix != -1:	
		ans_score = df_o.at[ans_ix,'ans']
		cnt = sum([1 for k in range(i,j) if df_o.at[k,'ans'] >= ans_score])
		mrr += 1.0/cnt
		list_cnt.append(cnt)
		cnt_q_have_ans += 1
	cnt_q += 1
	i = j

mrr /= cnt_q
print(mrr)	
print(cnt_q)
print(cnt_q_have_ans)
print(collections.Counter(list_cnt))
#%%保存
out_df = pd.DataFrame(preds[0])
out_df.to_csv("v{}.csv".format(mrr), index=False)

