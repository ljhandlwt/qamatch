# qamatch

## 文件

### 代码文件
lcs.c:最长公共子串<br/>
make_data_less.py:生成小数据集<br/>
make_test.py:给测试集加上假标签<br/>
make_wr.py:找出数据集中全部的疑问代词<br/>
utils.py:工具函数文件<br/>
word_seg.py:生成数据集的分词文件<br/>
make_feature.py:构造数据集的特征<br/>
xgb.py:xgb训练和测试<br/>
count_mrr.py:计算MRR<br/>

### 外部数据文件
word2vec.txt:词向量文件<br/>
wordsame.txt:同义词文件<br/>
stopword.txt:停用词文件<br/>

### 生成文件
wr_vocab.txt:疑问代词库<br/>
lcs.so:lcs链接库<br/>
train_seg.txt:数据集的分词文件<br/>
train_feature.txt:数据集的特征文件<br/>
.....<br/>

### 数据集
train.txt:训练集文件<br/>
train_less.txt:小训练集文件<br/>
dev.txt:开发集文件<br/>
test.txt:测试集文件<br/>

## 运行
0.获取必要的文件,并转换成对应的格式(参考utils.py中的读入)<br/>
1.用gcc编译c语言的库<br/>
2.执行make_data_less.py得到小数据集(可以不做)<br/>
3.执行make_test.py给测试集加上假标签(规范所有数据集的格式)<br/>
4.执行word_seg.py获取分词文件<br/>
5.执行make_wr.py获取疑问代词库(也可以外部得到)<br/>
6.执行make_feature.py构造特征文件(可以多次执行,逐步加特征)<br/>
7.执行xgb.py训练并测试<br/>
8.执行count_mrr.py得到MRR分数<br/>