# -*- coding: utf-8 -*-

import numpy as np 

# 口令中出现的字母表
alphabetas = [
	# 数字
	'1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 
	# 字母
	'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 
	'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 
	'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D',
	'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 
	# 特殊字符
	',', '.', '/', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', 
	')', '_', '+', '<', '>', '?', ' ', '{', '}', '|', ':', '\"', '[', 
	']', '\\', '\'', ';', '`', '-', '='
]

def char_to_vector(c, tables=alphabetas, start=False, end=False):
	"""
		convert character to 1-of-N code
	"""
	vec_len = len(tables) + 2
	vector = [ 0 for _ in range(vec_len) ]
	# start character
	if start:
		vector[0] = 1
		return vector
	# end character
	if end:
		vector[-1] = 1
		return vector

	vector[tables.index(c)+1] = 1
	return vector


def password_to_vector(password, width=3):
	if len(password) + 2 < width + 1:
		raise Exception('password is too short')

	X = []
	Y = []

	x0 = [ char_to_vector(None, start=True) ]
	for c in password[:width-1]:
		x0.append(char_to_vector(c))
	y0 = char_to_vector(password[width-1])
	X.append(x0)
	Y.append(y0)

	for i in range(len(password)-width):
		x = [ char_to_vector(c) for c in password[i:i+width] ]
		y = char_to_vector(password[i+width]) 
		X.append(x)
		Y.append(y)

	x1 = []
	for c in password[-width:]:
		x1.append(char_to_vector(c))
	y1 = char_to_vector(None, end=True)
	X.append(x1)
	Y.append(y1)

	return X, Y


def vector_to_char(vec, tables=alphabetas):
	# 结尾字符
	if vec[-1] == 1:
		return '△'
	# 开始字符
	if vec[0] == 1:
		return '□'

	for i, e in enumerate(vec):
		if e == 1:
			return tables[i-1]


def transform_dataset(seq_len):
	with open('../data/rockyou_small.txt', 'r') as f:
		X = []
		Y = []

		for i, pw in enumerate(f):
			try:
				x, y = password_to_vector(pw.rstrip(), width=seq_len)
			except Exception:
				print('[line %d] width < %d' % (i+1, seq_len))
			else:
				X += x
				Y += y

		X = np.array(X)
		Y = np.array(Y)
		print(X.shape, Y.shape)
		np.savez('dataset_%d'%seq_len, X=X, Y=Y)


def main():
	for l in range(1,7):
		transform_dataset(seq_len=l)

if __name__ == '__main__':
	main()

