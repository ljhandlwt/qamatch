import codecs
import os

##构造小训练集,用来调试程序

file = 'train.txt'

def main():
	file_base,file_ext = os.path.splitext(file)
	f = codecs.open(file, encoding='utf-8')
	lines = f.readlines()
	output_file = file_base + '_less' + file_ext
	with codecs.open(output_file, mode='w', encoding='utf-8') as fw:
		for i in range(min(5000,len(lines))):
			fw.write(lines[i])


if __name__ == '__main__':
	main()