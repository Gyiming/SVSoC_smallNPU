from __future__ import with_statement
import os
import configparser
import random
import numpy as np 
import pandas as pd 
import csv

def main():
	data1 = pd.read_csv('yolov2_Conv2_dram_filter_read.csv')
	#print(type(data1))
	x = np.array(data1)
	y = x[0:len(x),1:11]
	#print(y)
	t = y.flatten()
	a = t.shape
	b = a[0]
	#print(float(t[4][1:]))
	
	for i in range(1,b):
		if type(t[i]) == str:
			old_address = t[i-1]
			new_address = float(t[i][1:])
			if old_address == new_address -1.0:
				continue
			else:
				print new_address
		else:
			if type(t[i-1]) == str:
				old_address = float(t[i-1][1:])
				new_address = t[i]
				if old_address == new_address -1.0:
					continue
				else:
					print new_address
			else:
				old_address = t[i-1]
				new_address = t[i]
				if old_address == new_address -1.0:
					continue
				else:
					print new_address
	
	#print(t[9],t[8])

		#if (t[i-1] != t[i] - 1):
		#	print(t[i])
	#print(t)


	

if __name__ == '__main__':
	main()