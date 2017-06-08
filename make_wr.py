import codecs
import jieba
import jieba.posseg
import os

##找出所有代词,然后手动找出疑问代词

#files = ['test.txt']
files = ['dev_seg.txt','train_seg.txt']
#files = ['train_less_seg.txt']
output_file = 'wr_vocab.txt'

def main():
	vocab = set()
	if os.path.exists(output_file):
		f = codecs.open(file, encoding='utf-8')
		vocab = set([x.strip() for x in f.readline()])
	for file in files:
		print(file)
		f = codecs.open(file, encoding='utf-8')
		n = int(f.readline().strip())
		for i in range(n):
			label = f.readline()
			q_words = f.readline().strip().split('\t')
			q_flags = f.readline().strip().split('\t')
			a_words = f.readline()
			a_flags = f.readline()
			for j in range(len(q_words)):
				if q_flags[j] == 'r':
					vocab.add(q_words[j])
			print("{}/{}".format(i,n),end='\r')		

	with codecs.open(output_file, mode='w', encoding='utf-8') as fw:
		for word in vocab:
			fw.write(word+'\n')

if __name__ == '__main__':
	main()