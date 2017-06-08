#%%

import pandas as pd
import numpy as np
import math
import os
import codecs
import utils


## 构造正确的iawf特征


files = ['test.txt']
#files = ['dev.txt','train.txt']

#files = ['train_less.txt']
#%%



#iawf
def make_iawf(df_seg, df_feature):
	print("begin iawf")
	df_feature['iawf1'] = 0.0
	df_feature['iawf2'] = 0.0
	df_feature['iawf3'] = 0.0
	i = 0
	j = 0
	n = len(df_seg)
	cnt = 0
	while i < n:
		cnt += 1
		if cnt % 100 ==0 :
			print(cnt,end='\r')
		q_words = df_seg.at[i,'q_words']
		q_flags = df_seg.at[i,'q_flags']
		j = i + 1
		while j < n:
			if df_seg.loc[j,'q_words'] != q_words:
				break
			j += 1
		iawfs = [[0 for m in range(j-i)] for k in range(3)]
		wr_word = ""
		for iiwr in range(len(q_words)):
#                        print(q_flags[iiwr]+" "+q_words[iiwr])
			if(q_flags[iiwr] in ['ry','ryt','rys','ryv']):                           
				wr_word = q_words[iiwr]
#                            print(wr_word)
				break
	

		if wr_word != "":
			iwr = q_words.index(wr_word)
			for k in range(3):
				for m in range(2):
					curk = 0
					ix = iwr-1 if m==0 else iwr+1
					while curk < k+1 and ix>=0 and ix < len(q_words):
						q_flag = q_flags[ix]
						if q_flag[0] in ['n','v']:
							curk += 1 
						ix = ix - 1 if m==0 else ix+1
					
					if curk != k+1:
						continue

					ix = ix + 1 if m==0 else ix - 1
					q_word = q_words[ix]
					
					has_words = [0]*(j-i)
					cnt_words = [0]*(j-i)	
					for x in range(j-i):
						a_words = df_seg.at[x+i,'a_words']	
						for a_word in a_words:
							if q_word == a_word:
								has_words[x] = 1
								cnt_words[x] += 1
					cnt_sum_words = sum(cnt_words)
					cnt_words = [cnt_sum_words-x for x in cnt_words]
					for x in range(j-i):
						iawfs[k][x] += has_words[x]/(1.0+cnt_words[x])**3
		else:
			pass
			#print("[No wr]"+' '.join(q_words))					
		for k in range(i,j):
			df_feature.at[k,'iawf1'] = iawfs[0][k-i]
			df_feature.at[k,'iawf2'] = iawfs[1][k-i]
			df_feature.at[k,'iawf3'] = iawfs[2][k-i]
		i = j
	print("")	

#exiawf
def make_exiawf(df_seg, df_feature):
	print("begin ex-iawf")
	df_feature['exiawf1'] = 0.0
	df_feature['exiawf2'] = 0.0
	df_feature['exiawf3'] = 0.0
	i = 0
	j = 0
	n = len(df_seg)
	cnt = 0
	while i < n:
		cnt += 1
		if cnt % 100 ==0 :
			print(cnt,end='\r')
		q_words = df_seg.at[i,'q_words']
		q_flags = df_seg.at[i,'q_flags']
		j = i + 1
		while j < n:
			if df_seg.loc[j,'q_words'] != q_words:
				break
			j += 1
		iawfs = [[0 for m in range(j-i)] for k in range(3)]
		wr_word = ""
		for iiwr in range(len(q_words)):
#                        print(q_flags[iiwr]+" "+q_words[iiwr])
			if(q_flags[iiwr] in ['ry','ryt','rys','ryv']):                           
				wr_word = q_words[iiwr]
#                            print(wr_word)
				break
	

		if wr_word != "":
			iwr = q_words.index(wr_word)
			for k in range(3):
				for m in range(2):
					curk = 0
					ix = iwr-1 if m==0 else iwr+1
					while curk < k+1 and ix>=0 and ix < len(q_words):
						q_flag = q_flags[ix]
						if q_flag[0] in ['n','v']:
							curk += 1 
						ix = ix - 1 if m==0 else ix+1
					
					if curk != k+1:
						continue

					ix = ix + 1 if m==0 else ix - 1
					q_word = q_words[ix]
					
					has_words = [0]*(j-i)
					cnt_words = [0]*(j-i)	
					for x in range(j-i):
						a_words = df_seg.at[x+i,'a_words']	
						for a_word in a_words:
							if utils.is_same(q_word,a_word):
								has_words[x] = 1
								cnt_words[x] += 1
					cnt_sum_words = sum(cnt_words)
					cnt_words = [cnt_sum_words-x for x in cnt_words]
					for x in range(j-i):
						iawfs[k][x] += has_words[x]/(1.0+cnt_words[x])**3
		else:
			pass
			#print("[No wr]"+' '.join(q_words))					
		for k in range(i,j):
			df_feature.at[k,'exiawf1'] = iawfs[0][k-i]
			df_feature.at[k,'exiawf2'] = iawfs[1][k-i]
			df_feature.at[k,'exiawf3'] = iawfs[2][k-i]
		i = j
	print("")

#标签
def add_label(df, df_feature):
	df_feature['label'] = df['label']

def load_seg_file(file):
	file_base,file_ext = os.path.splitext(file)
	seg_file = file_base + '_seg' + file_ext	
	#f = codecs.open("D:\\pythonCode\\"+seg_file, 'r','utf-8')
	f = codecs.open(seg_file, encoding='utf-8')
	n = int(f.readline().strip())
	data = []
	for i in range(n):
		label = f.readline().strip()
		q_words = f.readline().strip().split('\t')
		q_flags = f.readline().strip().split('\t')
		a_words = f.readline().strip().split('\t')
		a_flags = f.readline().strip().split('\t')

		data.append([label, q_words, q_flags, a_words, a_flags])
		data[-1][0] = int(data[-1][0])
	df = pd.DataFrame(data)
	df.columns = ['label','q_words','q_flags','a_words','a_flags']	

	return df

def load_feature_file(file):
	file_base,file_ext = os.path.splitext(file)
	feature_file = file_base + '_feature' + file_ext
	if os.path.exists(feature_file):
		df = pd.read_csv(feature_file)
	else:
		df = pd.DataFrame()	
	return df	


def save_feature_file(file, df):
	file_base,file_ext = os.path.splitext(file)
	feature_file = file_base + '_feature' + file_ext
	df.to_csv(feature_file, index=False)

def main():
	#utils.load_word2vec('word2vec.txt')
	for file in files:
		print(file)
		#df = utils.load_file(file)
		df_seg = load_seg_file(file)
		df_feature = load_feature_file(file)

		make_iawf(df_seg, df_feature)
		make_exiawf(df_seg, df_feature)

		save_feature_file(file, df_feature)


if __name__ == "__main__":
	main()
 

#%%
