import codecs
import jieba
import jieba.posseg
import os

### discard
##构造数据集的分词文件

files = ['test.txt']
#files = ['dev.txt','train.txt']
#files = ['train_less.txt']

def main():
	for file in files:
		print(file)
		file_base,file_ext = os.path.splitext(file)
		f = codecs.open(file, encoding='utf-8')
		lines = f.readlines()
		output_file = file_base + '_seg' + file_ext
		with codecs.open(output_file, mode='w', encoding='utf-8') as fw:
			fw.write("{}\n".format(len(lines)))
			for i,line in enumerate(lines):
				ls = line.strip().split('\t')
				fw.write(ls[0]+'\n')
				question = list(jieba.posseg.cut(ls[1]))
				q_words = [p.word for p in question]
				q_flags = [p.flag for p in question]
				fw.write('\t'.join(q_words)+'\n')
				fw.write('\t'.join(q_flags)+'\n')
				answer = list(jieba.posseg.cut(ls[2]))
				a_words = [p.word for p in answer]
				a_flags = [p.flag for p in answer]
				fw.write('\t'.join(a_words)+'\n')
				fw.write('\t'.join(a_flags)+'\n')
				print('{}/{}'.format(i,len(lines)),end='\r')

if __name__ == '__main__':
	main()