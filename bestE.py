from __future__ import with_statement
import os
import random
import numpy as np 

def schedule(La,Lb,Lc,Ld,ast,bst,cst,dst):

	small = 1000000

	result = [La+ast,Lb+bst,Lc+cst,Ld+dst]

	for i in range(4):
		if result[i]<small:
			small = result[i]
			value = i

	return value

def enschedule(Lbudget,Ea,Eb,Ec,Ed,La,Lb,Lc,Ld):
	Lbudget = 900
	Ea = 10
	Eb = 69
	Ec = 58
	Ed = 319
	La = 100
	Lb = 186
	Lc = 231
	Ld = 565
	a=[0 for i in range(11)]
	b=[0 for i in range(11)]
	c=[0 for i in range(11)]
	d=[0 for i in range(11)]
	ft = [0 for i in range(11)]
	pk = [0 for i in range(11)]

	if (Lbudget<558):
		os._exit()
	elif (Lbudget<600):
		anum=5
		bnum=3
		cnum=2
	elif (Lbudget<700):
		anum=6
		bnum=2
		cnum=2
	elif (Lbudget<800):
		anum=7
		bnum=1
		cnum=2
	elif (Lbudget<900):
		anum=8
		cnum=2
		bnum=0
	elif (Lbudget<1000):
		anum=9
		cnum=1
		bnum=0
	else:
		anum=10
		bnum=0
		cnum=0

	for i in range(1,11):
		if (anum*bnum*cnum!=0):
			choice = schedule(La,Lb,Lc,Ld,a[i],b[i],c[i],d[i])
			if (choice==0):
				pk[i] = 'acc'
				ft[i] = a[i] + La
				for j in range(i+1,11):
					a[j] = ft[i]
					b[j] = b[i]
					c[j] = c[i]
					d[j] = d[i]
				anum = anum - 1
			if (choice==1):
				pk[i] = 'gpu'
				ft[i] = b[i] + Lb
				for j in range(i+1,11):
					a[j] = a[i]
					b[j] = b[i] + Lb
					c[j] = c[i] 
					d[j] = d[i]
				bnum = bnum-1
			if (choice==2):
				pk[i] = 'dsp'
				ft[i] = c[i] + Lc
				for j in range(i+1,11):
					a[j] = a[i]
					b[i] = b[i]
					c[j] = c[i] + Lc
					d[j] = d[i]
				cnum = cnum - 1

		elif (bnum==0):
			if (cnum==0):
				pk[i] = 'acc'
				ft[i] = a[i] + La
				for j in range(i+1,11):
					a[j] = ft[i]
					b[j] = b[i]
					c[j] = c[i]
					d[j] = d[i]
				anum = anum-1
			else:
				choice = schedule(La,10000,Lc,Ld,a[i],b[i],c[i],d[i])
				if (choice==0):
					pk[i] = 'acc'
					ft[i] = a[i] + La
					for j in range(i+1,11):
						a[j] = ft[i]
						b[j] = b[i]
						c[j] = c[i]
						d[j] = d[i]
					anum = anum - 1
				if (choice==1):
					pk[i] = 'gpu'
					ft[i] = b[i] + Lb
					for j in range(i+1,11):
						a[j] = a[i]
						b[i] = b[i] + Lbp
						c[j] = b[i] + Lb
						d[j] = d[i]
					bnum = bnum-1
				if (choice==2):
					pk[i] = 'dsp'
					ft[i] = c[i] + Lc
					for j in range(i+1,11):
						a[j] = a[i]
						b[i] = b[i]
						c[j] = c[i] + Lc
						d[j] = d[i]
					cnum = cnum - 1
		else:
			pk[i] = 'acc'
			ft[i] = a[i] + La
			for j in range(i+1,11):
				a[j] = ft[i]
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i]
			anum = anum - 1			

	return pk,ft

if __name__ == '__main__':
	main()