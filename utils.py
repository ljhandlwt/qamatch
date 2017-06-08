import ctypes
import numpy as np
import pandas as pd
import codecs
import os

##工具

lib_lcs = ctypes.CDLL('lcs.so')
dict_word2vec = None
ndim_word2vec = None
set_wr = None
dict_wordsame = None
set_stopword = None

#lcs
def make_carray(s):
	n = len(s)
	array = (ctypes.c_int*n)()
	for i in range(n):
		array[i] = ord(s[i])
	return array	

def lcs(s1,s2):
	c1 = make_carray(s1)
	c2 = make_carray(s2)
	return lib_lcs.lcs(c1,c2,len(s1),len(s2))

#word2vec
def word2vec(word):
	if dict_word2vec is None:
		load_word2vec('word2vec.txt')
	if word in dict_word2vec:
		return dict_word2vec[word]	
	else:
		return np.random.uniform(-0.1,0.1,ndim_word2vec)

def load_word2vec(file):
	print("load word2vec")
	f = codecs.open(file, encoding='utf-8')
	global dict_word2vec
	global ndim_word2vec
	n,ndim_word2vec = map(int,f.readline().strip().split())
	dict_word2vec = {}
	for i in range(n):
		line = f.readline()
		ls = line.strip().split()
		dict_word2vec[ls[0]] = [float(x) for x in ls[1:]]
		print("{}/{}".format(i,n),end='\r')
	print("")	

def save_word2vec(file):
	print("save word2vec")
	global dict_word2vec
	global ndim_word2vec
	with codecs.open(file, mode='w', encoding='utf-8') as f:
		n = len(dict_word2vec)
		f.write("{} {}\n".format(n,ndim_word2vec))
		for i,word in enumerate(dict_word2vec):
			vec = dict_word2vec[word]
			f.write("{} {}\n".format(word,' '.join(map(str,vec))))
			print("{}/{}".format(i,n),end='\r')
		print("")	

#疑问代词
def find_wr(words):
	for word in words:
		if is_wr(word):
			return word
	return ""

def is_wr(word):
	if set_wr is None:
		load_wr('wr_vocab.txt')
	return word in set_wr

def load_wr(file):
	f = codecs.open(file, encoding='utf-8')
	global set_wr
	set_wr = set([x.strip() for x in f.readlines()])

#同义词
def is_same(w1, w2):
	if dict_wordsame is None:
		load_wordsame('wordsame.txt')
	if w1 not in dict_wordsame or w2 not in dict_wordsame:
		return False
	return len(set(dict_wordsame[w1]).intersection(dict_wordsame[w2])) > 0

def load_wordsame(file):
	print("load wordsame")
	global dict_wordsame
	dict_wordsame = {}
	f = codecs.open(file, encoding='utf-8')
	data = f.readlines()
	for i,s in enumerate(data):
		g = s.strip().split()
		for word in g:
			if word not in dict_wordsame:
				dict_wordsame[word] = []
			dict_wordsame[word].append(i)
		print("{}/{}".format(i,len(data)),end='\r')
	print("")

#停用词
def is_stopword(word):
	if set_stopword is None:
		load_stopword('stopword.txt')
	return word in set_stopword

def load_stopword(file):
	print("load wordsame")
	global set_stopword
	f = codecs.open(file, encoding='utf-8')
	set_stopword = set([x.strip() for x in f.readlines()])

#数据集文件读入
def load_file(file):
	f = codecs.open(file, encoding='utf-8')
	data = []
	for line in f:
		line = line.strip().split('\t')
		data.append(line)
		data[-1][0] = int(data[-1][0])
	df = pd.DataFrame(data)
	df.columns = ['label','question','answer']	

	return df

def load_seg_file(file):
	file_base,file_ext = os.path.splitext(file)
	seg_file = file_base + '_seg' + file_ext	
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