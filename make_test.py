import codecs

## 给测试集加上假标签,方便统一格式

file = 'test.txt'

def main():
	print(file)
	f = codecs.open(file, encoding='utf-8')
	data = f.readlines()
	with codecs.open(file, mode='w', encoding='utf-8') as f:
		for line in data:
			f.write("0\t"+line)

if __name__ == '__main__':
	main()