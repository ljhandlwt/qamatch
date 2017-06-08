import utils
import pandas as pd
import numpy as np
import collections

##计算MRR(有bug)

df_o = pd.read_csv('v3.csv')
df = utils.load_file('dev.txt')

df_o.columns = ['ans']

mrr = 0
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
		cnt = sum([1 for k in range(i,j) if df_o.at[k,'ans'] > ans_score])
		cnt += 1
		mrr += 1/cnt
		list_cnt.append(cnt)
		cnt_q_have_ans += 1
	cnt_q += 1
	i = j

mrr /= cnt_q
print(mrr)	
print(cnt_q)
print(cnt_q_have_ans)
print(collections.Counter(list_cnt))