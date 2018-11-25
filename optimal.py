from __future__ import with_statement
import os
import configparser
import random
import numpy as np 
import schedule
import pandas as pd 

def recursive(l,latency_ACC,latency_GPU,latency_DSP,latency_CPU,energy_ACC,energy_GPU,energy_DSP,energy_CPU,frame):
	if len(l) == frame:
		acc_time = 0
		gpu_time = 0
		dsp_time = 0
		cpu_time = 0
		finish_time = 0
		total_energy = 0
		result = ''
		for j in range(frame):
			if (l[j] == 1):
				acc_time = acc_time + 1
				finish_time = finish_time + acc_time*latency_ACC
				result = result + 'acc' + ','
				total_energy = total_energy + energy_ACC
			if (l[j] == 2):
				gpu_time = gpu_time + 1
				finish_time = finish_time + gpu_time*latency_GPU
				result = result + 'gpu' + ','
				total_energy = total_energy + energy_GPU
			if (l[j] == 3):
				dsp_time = dsp_time + 1
				finish_time = finish_time + dsp_time*latency_DSP
				result = result + 'dsp' + ','
				total_energy = total_energy + energy_DSP
			if (l[j] == 4):
				cpu_time = cpu_time + 1
				finish_time = finish_time + cpu_time*latency_CPU
				result = result + 'cpu' + ','
				total_energy = total_energy + energy_CPU
		result = result + str(finish_time) + ',' + str(total_energy)
		print(result)
		f = open('test.txt','a')
		f.write(result+'\n')
		f.close()
		return 
	for i in range(1,5):
		recursive(l+[i],latency_ACC,latency_GPU,latency_DSP,latency_CPU,energy_ACC,energy_GPU,energy_DSP,energy_CPU,frame)

def main():
	acc_time = 0
	gpu_time = 0
	dsp_time = 0
	cpu_time = 0
	finish_time = 0
	result = ''
	total_energy = 0
	config=configparser.ConfigParser()
	mini = 10000000
	frame = 10
	energy_budget = 500
	with open("soc_configure.cfg","r+") as cfgfile:
		config.readfp(cfgfile)
        latency_ACC = int(config.get("info","latency_accelerator"))
        latency_GPU = int(config.get("info","latency_GPU"))
        latency_DSP = int(config.get("info","latency_DSP"))
        latency_CPU = int(config.get("info","latency_CPU1"))
        energy_ACC = int(config.get("info","energy_accelerator"))
        energy_CPU = int(config.get("info","energy_CPU1"))
        energy_GPU = int(config.get("info","energy_GPU"))
        energy_DSP = int(config.get("info","energy_DSP"))

	recursive([],latency_ACC,latency_GPU,latency_DSP,latency_CPU,energy_ACC,energy_GPU,energy_DSP,energy_CPU,frame)
	f = open('test.txt','r')
	content = f.readlines()
	f.close()

	for i in range(0,len(content)):
		a = content[i].split(',')
		a[frame] = int(a[frame])
		a[frame+1] = int(a[frame+1].rstrip('\n'))
		if a[frame+1] > energy_budget:
			continue
		if a[frame] < mini:
			mini = a[frame]
	print(mini)
	pk,tk= schedule.laschdule(energy_budget,energy_ACC,energy_GPU,energy_DSP,energy_CPU,1000000,latency_ACC,latency_GPU,latency_DSP,latency_CPU,1000000,frame)
	print(pk)
	for j in range(1,len(pk)):			
		if (pk[j] == 'acc1'):
			acc_time = acc_time + 1
			finish_time = finish_time + acc_time*latency_ACC
			result = result + 'acc' + ','
			total_energy = total_energy + energy_ACC
		if (pk[j] == 'gpu'):
			gpu_time = gpu_time + 1
			finish_time = finish_time + gpu_time*latency_GPU
			result = result + 'gpu' + ','
			total_energy = total_energy + energy_GPU
		if (pk[j] == 'dsp'):
			dsp_time = dsp_time + 1
			finish_time = finish_time + dsp_time*latency_DSP
			result = result + 'dsp' + ','
			total_energy = total_energy + energy_DSP
		if (pk[j] == 'cpu'):
			cpu_time = cpu_time + 1
			finish_time = finish_time + cpu_time*latency_CPU
			result = result + 'cpu' + ','
			total_energy = total_energy + energy_CPU
	print(finish_time)





if __name__ == '__main__':
	main()