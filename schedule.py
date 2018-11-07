from __future__ import with_statement
import os
import random
import numpy as np 

def check(i,Ebudget,Ea,Eb,Ec,Ed,aenable,benable,cenable,denable):
	aenable=1
	benable=1
	cenable=1
	denable=1
	if (((Ebudget-Ea)/Ea)<(10-i) and ((Ebudget-Ea)/Eb)<(10-i) and ((Ebudget-Ea)/Ec)<(10-i) and ((Ebudget-Ea)/Ed)<(10-i)):
		aenable = 0
	if (((Ebudget-Eb)/Ea)<(10-i) and ((Ebudget-Eb)/Eb)<(10-i) and ((Ebudget-Eb)/Ec)<(10-i) and ((Ebudget-Eb)/Ed)<(10-i)):
		benable = 0
	if (((Ebudget-Ec)/Ea)<(10-i) and ((Ebudget-Ec)/Eb)<(10-i) and ((Ebudget-Ec)/Ec)<(10-i) and ((Ebudget-Ed)/Ed)<(10-i)):
		cenable = 0
	if (((Ebudget-Ed)/Ea)<(10-i) and ((Ebudget-Ed)/Ea)<(10-i) and ((Ebudget-Ed)/Ec)<(10-i) and ((Ebudget-Ed)/Ed)<(10-i)):
		denable = 0
	return aenable,benable,cenable,denable

	


def schecule(i,aenable,benable,cenable,denable,La,Lb,Lc,Ld,Ea,Eb,Ec,Ed,Ebudget,ast,bst,cst,dst):
	aenable,benable,cenable,denable=check(i,Ebudget,Ea,Eb,Ec,Ed,aenable,benable,cenable,denable)

	if (aenable==0):
		ast=10000
	if (benable==0):
		bst=10000
	if (cenable==0):
		cst=10000
	if (denable==0):
		dst=10000

	small = 1000000

	result = [La+ast,Lb+bst,Lc+cst,Ld+dst]

	for i in range(4):
		if result[i]<small:
			small = result[i]
			value = i

	return value


def laschdule(Ebudget,Ea,Eb,Ec,Ed,La,Lb,Lc,Ld):
	'''
	Ebudget = 400
	Ea = 10
	Eb = 69
	Ec = 58
	Ed = 319
	La = 100
	Lb = 186
	Lc = 231
	Ld = 565
	'''
	a=[0 for i in range(11)]
	b=[0 for i in range(11)]
	c=[0 for i in range(11)]
	d=[0 for i in range(11)]
	ft = [0 for i in range(11)]
	pk = [0 for i in range(11)]
	aenable = [0 for i in range(11)]
	benable = [0 for i in range(11)]
	cenable = [0 for i in range(11)]
	denable = [0 for i in range(11)]

	anum = Ebudget/Ea
	bnum = Ebudget/Eb
	cnum = Ebudget/Ec
	dnum = Ebudget/Ed

	if (anum<10):
		print('unable to achieve')
		os._exit()

	for i in range(1,11):
		if (anum>0):
			aenable[i] = 1
		if (bnum>0):
			benable[i] = 1
		if (cnum>0):
			cenable[i] = 1
		if (dnum>0):
			denable[i] = 1
		choice = schecule(i,aenable[i],benable[i],cenable[i],denable[i],La,Lb,Lc,Ld,Ea,Eb,Ec,Ed,Ebudget,a[i],b[i],c[i],d[i])
		if (choice== 0):
			pk[i] = 'acc'
			ft[i] = a[i] + La
			for j in range(i+1,11):
				a[j] = ft[i]
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i]
			Ebudget = Ebudget - Ea

		if (choice == 1):
			pk[i] = 'gpu'
			ft[i] = b[i] + Lb
			for j in range(i+1,11):
				a[j] = a[i]
				b[j] = b[i] + Lb
				c[j] = c[i]
				d[j] = d[i]
			Ebudget = Ebudget - Eb

		if (choice == 2):
			pk[i] = 'dsp'
			ft[i] = c[i] + Lc
			for j in range(i+1,11):
				a[j] = a[i]
				b[i] = b[i]
				c[j] = c[i] + Lc
				d[j] = d[i]
			Ebudget = Ebudget - Ec

		if (choice == 3):
			pk[i] = 'cpu'
			ft[i] = d[i] + Ld
			for j in range(i+1,11):
				a[j] = a[i]
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i] + Lde
			Ebudget = Ebudget - Ed

	#for i in range(1,11):
	#	print(ft[i])

	return pk,ft

if __name__ == '__main__':
	main()


