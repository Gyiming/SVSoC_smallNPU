from __future__ import with_statement
import os
import random
import numpy as np 

def check(i,Ebudget,Ea,Eb,Ec,Ed,Ee,aenable,benable,cenable,denable,eenable,k):
	aenable=1
	benable=1
	cenable=1
	denable=1
	eenable=1
	if (((Ebudget-Ea)/Ea)<(k-i) and ((Ebudget-Ea)/Eb)<(k-i) and ((Ebudget-Ea)/Ec)<(k-i) and ((Ebudget-Ea)/Ed)<(k-i) and ((Ebudget-Ea)/Ee)<(k-i)):
		aenable = 0
	if (((Ebudget-Eb)/Ea)<(k-i) and ((Ebudget-Eb)/Eb)<(k-i) and ((Ebudget-Eb)/Ec)<(k-i) and ((Ebudget-Eb)/Ed)<(k-i) and ((Ebudget-Eb)/Ee)<(k-i)):
		benable = 0
	if (((Ebudget-Ec)/Ea)<(k-i) and ((Ebudget-Ec)/Eb)<(k-i) and ((Ebudget-Ec)/Ec)<(k-i) and ((Ebudget-Ed)/Ed)<(k-i) and ((Ebudget-Ec)/Ee)<(k-i)):
		cenable = 0
	if (((Ebudget-Ed)/Ea)<(k-i) and ((Ebudget-Ed)/Eb)<(k-i) and ((Ebudget-Ed)/Ec)<(k-i) and ((Ebudget-Ed)/Ed)<(k-i) and ((Ebudget-Ed)/Ee)<(k-i)):
		denable = 0
	if (((Ebudget-Ee)/Ea)<(k-i) and ((Ebudget-Ee)/Eb)<(k-i) and ((Ebudget-Ee)/Ec)<(k-i) and ((Ebudget-Ee)/Ed)<(k-i) and ((Ebudget-Ee)/Ee)<(k-i)):
		eenable = 0

	return aenable,benable,cenable,denable,eenable

	
def schecule(i,aenable,benable,cenable,denable,eenable,La,Lb,Lc,Ld,Le,Ea,Eb,Ec,Ed,Ee,Ebudget,ast,bst,cst,dst,est,k):
	aenable,benable,cenable,denable,eenable=check(i,Ebudget,Ea,Eb,Ec,Ed,Ee,aenable,benable,cenable,denable,eenable,k)

	if (aenable==0):
		ast=10000
	if (benable==0):
		bst=10000
	if (cenable==0):
		cst=10000
	if (denable==0):
		dst=10000
	if (eenable==0):
		est=10000

	small = 1000000

	result = [La+ast,Lb+bst,Lc+cst,Ld+dst,Le+est]

	for i in range(5):
		if result[i]<small:
			small = result[i]
			value = i

	return value


def laschdule(Ebudget,Ea,Eb,Ec,Ed,Ee,La,Lb,Lc,Ld,Le,k):
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
	a=[0 for i in range(k+1)]
	b=[0 for i in range(k+1)]
	c=[0 for i in range(k+1)]
	d=[0 for i in range(k+1)]
	e=[0 for i in range(k+1)]
	ft = [0 for i in range(k+1)]
	pk = [0 for i in range(k+1)]
	aenable = [0 for i in range(k+1)]
	benable = [0 for i in range(k+1)]
	cenable = [0 for i in range(k+1)]
	denable = [0 for i in range(k+1)]
	eenable = [0 for i in range(k+1)]

	anum = Ebudget/Ea
	bnum = Ebudget/Eb
	cnum = Ebudget/Ec
	dnum = Ebudget/Ed
	enum = Ebudget/Ee 

	if (anum<k):
		print('unable to achieve')
		os._exit()

	for i in range(1,k+1):
		if (anum>0):
			aenable[i] = 1
		if (bnum>0):
			benable[i] = 1
		if (cnum>0):
			cenable[i] = 1
		if (dnum>0):
			denable[i] = 1
		if (enum>0):
			eenable[i] = 1
		choice = schecule(i,aenable[i],benable[i],cenable[i],denable[i],eenable[i],La,Lb,Lc,Ld,Le,Ea,Eb,Ec,Ed,Ee,Ebudget,a[i],b[i],c[i],d[i],e[i],k)
		if (choice== 0):
			pk[i] = 'acc1'
			ft[i] = a[i] + La
			for j in range(i+1,k+1):
				a[j] = a[i] + La
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i]
				e[j] = e[i]
			Ebudget = Ebudget - Ea

		if (choice == 1):
			pk[i] = 'gpu'
			ft[i] = b[i] + Lb
			for j in range(i+1,k+1):
				a[j] = a[i]
				b[j] = b[i] + Lb
				c[j] = c[i]
				d[j] = d[i]
				e[j] = e[i]
			Ebudget = Ebudget - Eb

		if (choice == 2):
			pk[i] = 'dsp'
			ft[i] = c[i] + Lc
			for j in range(i+1,k+1):
				a[j] = a[i]
				b[i] = b[i]
				c[j] = c[i] + Lc
				d[j] = d[i]
				e[j] = e[i]
			Ebudget = Ebudget - Ec

		if (choice == 3):
			pk[i] = 'cpu'
			ft[i] = d[i] + Ld
			for j in range(i+1,k+1):
				a[j] = a[i]
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i] + Ld
				e[j] = e[i]
			Ebudget = Ebudget - Ed

		if (choice == 4):
			pk[i] = 'acc2'
			ft[i] = e[i] + Le
			for j in range(i+1,k+1):
				a[j] = a[i]
				b[j] = b[i]
				c[j] = c[i]
				d[j] = d[i]
				e[j] = e[i] + Le
			Ebudget = Ebudget - Ee		

	#for i in range(1,11):
	#	print(ft[i])

	return pk,ft

def main():
	pk,ft = laschdule(500,11,134,114,319,11,344,619,743,1000,344,10)
	print sum(ft)


if __name__ == '__main__':
	main()


