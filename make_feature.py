import utils
import os
import pandas as pd
import numpy as np
import math

## 构造特征

files = ['test.txt']
#files = ['dev.txt','train.txt']
#files = ['train_less.txt']

#lcs
def make_lcs(df,df_feature):
	print("begin lcs")
	df_feature['lcs'] = df.apply(lambda x:utils.lcs(x['question'],x['answer']), axis=1)
	df_feature['len_q'] = df.apply(lambda x:len(x['question']), axis=1)
	df_feature['lcs_p'] = df_feature.apply(lambda x:x['lcs']/x['len_q'], axis=1)
	df_feature.drop('len_q', axis=1, inplace=True)

#词覆盖
def make_overlap(df_seg, df_feature):
	print("begin overlap")
	df_feature['overlap'] = df_seg.apply(lambda x:len(set(x['q_words']).intersection(x['a_words'])), axis=1)
	df_feature['len_q'] = df_seg.apply(lambda x:len(x['q_words']), axis=1)
	df_feature['overlap_p'] = df_feature.apply(lambda x:x['overlap']/x['len_q'], axis=1)
	df_feature.drop('len_q', axis=1, inplace=True)

#去掉停用词的词覆盖
def filter_stopword(words):
	return list(filter(lambda x:not utils.is_stopword(x),words))

def make_overlap_stoplist(df_seg, df_feature):
	print("begin overlap with filter stop words")
	df_feature['overlap_sl'] = df_seg.apply(lambda x:len(set(filter_stopword(x['q_words'])).intersection(filter_stopword(x['a_words']))), axis=1)
	df_feature['len_q'] = df_seg.apply(lambda x:len(filter_stopword(x['q_words'])), axis=1)
	df_feature['overlap_p_sl'] = df_feature.apply(lambda x:x['overlap']/x['len_q'] if x['len_q']!=0 else 0, axis=1)
	df_feature.drop('len_q', axis=1, inplace=True)

#词袋向量距离
def make_bag_distance(df_seg, df_feature):
	print("begin bag distance")
	nrow = len(df_seg)
	dist_cos = [0]*nrow
	dist_jac = [0]*nrow
	dist_man = [0]*nrow
	for index,row in df_seg.iterrows():
		q_words = set(row['q_words'])
		a_words = set(row['a_words'])
		union_words = q_words.union(a_words)
		q_vec = [0]*len(union_words)
		a_vec = [0]*len(union_words)
		for i,word in enumerate(union_words):
			q_vec[i] = 1 if word in q_words else 0
			a_vec[i] = 1 if word in a_words else 0
		s_dot = sum([qi*ai for qi,ai in zip(q_vec,a_vec)])
		s_qlen = math.sqrt(sum(q_vec))
		s_alen = math.sqrt(sum(a_vec))
		s_union = sum([qi|ai for qi,ai in zip(q_vec,a_vec)])
		s_inter = sum([qi&ai for qi,ai in zip(q_vec,a_vec)])
		s_xor = sum([qi^ai for qi,ai in zip(q_vec,a_vec)])
		d_cos = s_dot/(s_qlen*s_alen)
		d_jac = (s_union-s_inter)/s_union
		d_man = s_xor
		dist_cos[index] = d_cos
		dist_jac[index] = d_jac
		dist_man[index] = d_man
		print("{}/{}".format(index,nrow),end='\r')
	df_feature['dist_bag_cos'] = dist_cos
	df_feature['dist_bag_jac'] = dist_jac
	df_feature['dist_bag_man'] = dist_man

#词向量距离
def make_vec_distance(df_seg, df_feature):
	print("begin vec distance")
	nrow = len(df_seg)
	dist_cos = [0]*nrow
	dist_man = [0]*nrow
	dist_euc = [0]*nrow
	for index,row in df_seg.iterrows():
		q_words = row['q_words']
		a_words = row['a_words']
		q_vec = []
		a_vec = []
		for word in q_words:
			q_vec.append(utils.word2vec(word))
		for word in a_words:
			a_vec.append(utils.word2vec(word))	
		q_vec = np.array(q_vec).sum(axis=0) / len(q_words) 
		a_vec = np.array(a_vec).sum(axis=0)	/ len(a_words)
		s_dot = (q_vec*a_vec).sum()
		s_qlen = math.sqrt((q_vec**2).sum())
		s_alen = math.sqrt((a_vec**2).sum())
		s_dist = (np.abs(q_vec-a_vec)).sum()
		s_euc = math.sqrt(((q_vec-a_vec)**2).sum())
		d_cos = s_dot/(s_qlen*s_alen)
		d_man = s_dist
		d_euc = s_euc
		dist_cos[index] = d_cos
		dist_man[index] = d_man
		dist_euc[index] = d_euc
		print("{}/{}".format(index,nrow),end='\r')
	df_feature['dist_vec_cos'] = dist_cos
	df_feature['dist_vec_man'] = dist_man
	df_feature['dist_vec_euc'] = dist_euc

#iawf
def make_iawf(df_seg, df_feature):
	print("begin iawf")
	df_feature['old_iawf1'] = 0.0
	df_feature['old_iawf2'] = 0.0
	df_feature['old_iawf3'] = 0.0
	i = 0
	j = 0
	n = len(df_seg)
	while i < n:
		q_words = df_seg.at[i,'q_words']
		q_flags = df_seg.at[i,'q_flags']
		j = i + 1
		while j < n:
			if df_seg.loc[j,'q_words'] != q_words:
				break
			j += 1
		iawfs = [[0 for m in range(j-i)] for k in range(3)]
		wr_word = utils.find_wr(q_words)
		if wr_word == "":
			for word in q_words:
				if word.find('多') != -1 or word.find('几') != -1 or word.find('是否') != -1:
					wr_word = word
					break
		if wr_word != "":
			iwr = q_words.index(wr_word)
			for k in range(3):
				for m in range(2):
					ix = iwr-(k+1) if m==0 else iwr+(k+1)
					if not 0<=ix<len(q_words):
						continue
					q_word = q_words[ix]
					q_flag = q_flags[ix]
					if q_flag[0] not in ['n','v']:
						continue
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
						iawfs[k][x] += has_words[x]/(1+cnt_words[x])**3
		else:
			pass
			#print("[No wr]"+' '.join(q_words))					
		for k in range(i,j):
			df_feature.at[k,'old_iawf1'] = iawfs[0][k-i]
			df_feature.at[k,'old_iawf2'] = iawfs[1][k-i]
			df_feature.at[k,'old_iawf3'] = iawfs[2][k-i]
		i = j
		print("{}/{}".format(i,n),end='\r')
	print("")	

#ex-iawf
def make_exiawf(df_seg, df_feature):
	print("begin exiawf")
	df_feature['old_exiawf1'] = 0.0
	df_feature['old_exiawf2'] = 0.0
	df_feature['old_exiawf3'] = 0.0
	i = 0
	j = 0
	n = len(df_seg)
	while i < n:
		q_words = df_seg.at[i,'q_words']
		q_flags = df_seg.at[i,'q_flags']
		j = i + 1
		while j < n:
			if df_seg.at[j,'q_words'] != q_words:
				break
			j += 1
		iawfs = [[0 for m in range(j-i)] for k in range(3)]
		wr_word = utils.find_wr(q_words)
		if wr_word == "":
			for word in q_words:
				if word.find('多') != -1 or word.find('几') != -1 or word.find('是否') != -1:
					wr_word = word
					break
		if wr_word != "":
			iwr = q_words.index(wr_word)
			for k in range(3):
				for m in range(2):
					ix = iwr-(k+1) if m==0 else iwr+(k+1)
					if not 0<=ix<len(q_words):
						continue
					q_word = q_words[ix]
					q_flag = q_flags[ix]
					if q_flag[0] not in ['n','v']:
						continue
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
						iawfs[k][x] += has_words[x]/(1+cnt_words[x])**3	
		else:
			pass
			#print("[No wr]"+' '.join(q_words))							
		for k in range(i,j):
			df_feature.at[k,'old_exiawf1'] = iawfs[0][k-i]
			df_feature.at[k,'old_exiawf2'] = iawfs[1][k-i]
			df_feature.at[k,'old_exiawf3'] = iawfs[2][k-i]
		i = j
		print("{}/{}".format(i,n),end='\r')
	print("")

#标签
def add_label(df, df_feature):
	df_feature['label'] = df['label']


def main():
	#utils.load_word2vec('word2vec.txt')
	for file in files:
		print(file)
		#df = utils.load_file(file)
		df_seg = utils.load_seg_file(file)
		df_feature = utils.load_feature_file(file)

		#make_iawf(df_seg, df_feature)
		make_exiawf(df_seg, df_feature)

		utils.save_feature_file(file, df_feature)


if __name__ == "__main__":
	main()