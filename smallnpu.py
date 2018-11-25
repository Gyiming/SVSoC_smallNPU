# -* - coding: UTF-8 -* -
from __future__ import with_statement
from __future__ import print_function
import os
import configparser
import random
import numpy as np 
import schedule
import bestE


def main():
    config=configparser.ConfigParser()
    enable_performance = [0 for i in range(11)]
    enable_energy = [0 for i in range(11)]
    start_time = [0 for i in range(10000)]
    sensing_time = [0 for i in range(10000)]
    ISP_time = [0 for i in range(10000)]
    accelerator_time = [0 for i in range(10000)]
    smallnpu = [0 for i in range(10000)]
    predict_time = [0 for i in range(10000)]
    accumulation_accelerator = 0
    accumulation_smallnpu = 0
    total_energy_accelerator = 0
    total_energy_smallnpu = 0
    small_flag = 0
    snpu1 = 0
    snpu2 = 0
    ssim = np.load('ssim.npy')
    ssim_real = [0 for i in range(826)]
    for i in range(1,825):
        ssim_real[i] = ssim[i-1]
    with open("soc_configure.cfg","r+") as cfgfile:
        config.readfp(cfgfile)
        frame = int(config.get("info","frame"))
        frames_predicted = int(config.get("info","frames_predicted"))
        frames_checked = int(config.get("info","frames_checked"))
        latency_sensing = int(config.get("info","latency_sensing"))
        latency_ISP = int(config.get("info","latency_ISP"))
        latency_CPU1 = int(config.get("info","latency_CPU1"))
        latency_CPU2 = int(config.get("info","latency_CPU2"))
        latency_CPU3 = int(config.get("info","latency_CPU3"))
        latency_CPU4 = int(config.get("info","latency_CPU4"))
        latency_CPU5 = int(config.get("info","latency_CPU5"))
        latency_CPU6 = int(config.get("info","latency_CPU6"))
        latency_CPU7 = int(config.get("info","latency_CPU7"))
        latency_CPU8 = int(config.get("info","latency_CPU8"))
        latency_GPU = int(config.get("info","latency_GPU"))
        latency_DSP = int(config.get("info","latency_DSP"))
        latency_accelerator = int(config.get("info","latency_accelerator"))
        latency_predict = int(config.get("info","latency_predict"))
        latency_check = int(config.get("info","latency_check"))
        latency_commit = int(config.get("info","latency_commit"))
        latency_smallnpu = int(config.get("info","latency_smallnpu"))
        energy_sensing = int(config.get("info","energy_sensing"))
        energy_ISP = int(config.get("info","energy_ISP"))
        energy_CPU1 = int(config.get("info","energy_CPU1"))
        energy_CPU2 = int(config.get("info","energy_CPU2"))
        energy_CPU3 = int(config.get("info","energy_CPU3"))
        energy_CPU4 = int(config.get("info","energy_CPU4"))
        energy_CPU5 = int(config.get("info","energy_CPU5"))
        energy_CPU6 = int(config.get("info","energy_CPU6"))
        energy_CPU7 = int(config.get("info","energy_CPU7"))
        energy_CPU8 = int(config.get("info","energy_CPU8"))
        energy_GPU = int(config.get("info","energy_GPU"))
        energy_DSP = int(config.get("info","energy_DSP"))
        energy_accelerator = int(config.get("info","energy_accelerator"))
        energy_predict = int(config.get("info","energy_predict"))
        energy_check = int(config.get("info","energy_check"))
        energy_commit = int(config.get("info","energy_commit"))
        energy_smallnpu = int(config.get("info","energy_smallnpu"))
        accuracy = float(config.get("info","accuracy"))
        energy_budget = float(config.get("info","energy_budget"))
        latency_budget = float(config.get("info","latency_budget"))
        app_degree = int(config.get("info","approximation_degree"))



    for i in range(1, frame+1):
        fuck = 0
    	if (i==1):
    	    predict_frame_location=1;
    	    start_time[i] = 0
            #sensing_time
            sensing_time[i] = start_time[i] + latency_sensing
            total_energy_accelerator = total_energy_accelerator + energy_sensing
            total_energy_smallnpu = total_energy_smallnpu + energy_sensing
            #ISP_time
            ISP_time[i] = sensing_time[i] + latency_ISP
            total_energy_accelerator = total_energy_accelerator + energy_ISP
            total_energy_smallnpu = total_energy_smallnpu + energy_ISP
            #accelerator baseline
            accelerator_time[i] = ISP_time[i] + latency_accelerator + latency_commit
            total_energy_accelerator = total_energy_accelerator + energy_accelerator + energy_commit
            accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
            #smallnpu baseline
            smallnpu[i] = ISP_time[i] + latency_smallnpu + latency_commit
            total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
            accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]
            snpu1 = smallnpu[i] - latency_commit
            small_flag = 1
            snpu2 = snpu2
            #predict time
            for j in range(2,frames_predicted+1):
                predict_time[j] = ISP_time[i] + latency_predict


        elif ((i-predict_frame_location) == (frames_predicted+1)):
            predict_frame_location = i
            start_time[i] = sensing_time[i-1]
            #sensing
            sensing_time[i] = start_time[i] + latency_sensing
            total_energy_accelerator = total_energy_accelerator + energy_sensing
            total_energy_smallnpu = total_energy_smallnpu + energy_sensing
            #ISP
            if (ISP_time[i-1] > sensing_time[i]):
                ISP_time[i] = ISP_time[i-1] + latency_ISP
            else:
                ISP_time[i] = sensing_time[i] + latency_ISP
            total_energy_accelerator = total_energy_accelerator + energy_ISP
            total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
            #accelerator baseline
            if (accelerator_time[i-1] > ISP_time[i]):
                accelerator_time[i] = accelerator_time[i-1] + latency_accelerator + latency_commit
            else:
                accelerator_time[i] = ISP_time[i] + latency_accelerator + latency_commit
            total_energy_accelerator = total_energy_accelerator + energy_accelerator
            accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
            #smallnpu 
            if (small_flag==1):
                if (snpu2 > ISP_time[i]):
                    smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                    snpu2 = smallnpu[i] - latency_commit
                    snpu1 = snpu1
                    small_flag = 2
                else:
                    smallnpu[i] = ISP_time[i] + latency_smallnpu + latency_commit
                    snpu2 = smallnpu[i] - latency_commit
                    snpu1 = snpu1
                    small_flag = 2
            elif (small_flag==2):
                if snpu1 > ISP_time[i]:
                    smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                    snpu1 = smallnpu[i] - latency_commit
                    snpu2 = snpu2
                    small_flag = 1
                else:
                    smallnpu[i] = ISP_time[i] + latency_smallnpu + latency_commit
                    snpu1 = smallnpu[i] - latency_commit
                    snpu2 = snpu2
                    small_flag = 1
            total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
            accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]
            #predict time                
            for j in range(i+1,i+frames_predicted+1):
                predict_time[j] = ISP_time[i] + latency_predict

        else:
            if (i<=app_degree):
                total_energy_smallnpu = total_energy_smallnpu + energy_check
                if (ssim_real[i] < accuracy):
                    start_time[i] = sensing_time[i-1]
                    #sensing
                    sensing_time[i] = start_time[i] + latency_sensing
                    total_energy_accelerator = total_energy_accelerator + energy_sensing
                    total_energy_smallnpu = total_energy_smallnpu + energy_sensing
                    #ISP
                    if (ISP_time[i-1] > sensing_time[i]):
                        ISP_time[i] = ISP_time[i-1] + latency_ISP
                    else:
                        ISP_time[i] = sensing_time[i] + latency_ISP
                    total_energy_accelerator = total_energy_accelerator + energy_ISP
                    total_energy_smallnpu = total_energy_smallnpu + energy_ISP
                    #accelerator baseline
                    if (accelerator_time[i-1] > ISP_time[i]):
                        accelerator_time[i] = accelerator_time[i-1] + latency_accelerator
                    else:
                        accelerator_time[i] = ISP_time[i] + latency_accelerator
                    total_energy_accelerator = total_energy_accelerator + energy_accelerator
                    accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
                    #smallnpu:
                    if (i-predict_frame_location==1):
                        if (ISP_time[i] + latency_check) > predict_time[i]:
                            if (small_flag==1):
                                if snpu2 >(ISP_time[i] + latency_check):
                                    smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                                    small_flag = 2
                                    snpu1 = snpu1
                                    snpu2 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                    small_flag = 2
                                    snpu1 = snpu1
                                    snpu2 = smallnpu[i] - latency_commit
                            elif (small_flag == 2):
                                if snpu1 >(ISP_time[i] + latency_check):
                                    smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                                    small_flag = 1
                                    snpu2 = snpu2
                                    snpu1 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                    small_flag = 1
                                    snpu2 = snpu2
                                    snpu1 = smallnpu[i] - latency_commit                                
                        else:
                            if (small_flag==1):
                                if snpu2 >predict_time[i]:
                                    smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                                    small_flag = 2
                                    snpu1 = snpu1
                                    snpu2 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = predict_time[i] + latency_smallnpu
                                    small_flag = 2
                                    snpu1 = snpu1
                                    snpu2 = smallnpu[i] - latency_commit
                            elif (small_flag == 2):
                                if snpu1 >predict_time[i]:
                                    smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                                    small_flag = 1
                                    snpu2 = snpu2
                                    snpu1 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = predict_time[i] + latency_smallnpu
                                    small_flag = 1
                                    snpu2 = snpu2
                                    snpu1 = smallnpu[i] - latency_commit 
                    else:
                        if (small_flag==1):
                            if (ISP_time[i] + latency_check) > snpu2:
                                smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                snpu1 = snpu1
                                small_flag = 2
                                snpu2 = smallnpu[i] - latency_commit
                            else:
                                smallnpu[i] = snpu2 + latency_smallnpu
                                snpu1 = snpu1
                                small_flag = 2
                                snpu2 = smallnpu[i] - latency_commit                                
                        elif (small_flag==2):
                            if (ISP_time[i] + latency_check) > snpu1:
                                smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                snpu2 = snpu2
                                small_flag = 2
                                snpu1 = smallnpu[i] - latency_commit
                            else:
                                smallnpu[i] = snpu1 + latency_smallnpu
                                snpu2 = snpu2
                                small_flag = 2
                                snpu1 = smallnpu[i] - latency_commit         
                    total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
                    accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]

                else:
                    start_time[i] = sensing_time[i-1]
                    #sensing
                    sensing_time[i] = start_time[i] + latency_sensing
                    total_energy_accelerator = total_energy_accelerator + energy_sensing
                    total_energy_smallnpu = total_energy_smallnpu + energy_sensing
                    #ISP
                    if (ISP_time[i-1] > sensing_time[i]):
                        ISP_time[i] = ISP_time[i-1] + latency_ISP
                    else:
                        ISP_time[i] = sensing_time[i] + latency_ISP
                    total_energy_accelerator = total_energy_accelerator + energy_ISP
                    total_energy_smallnpu = total_energy_smallnpu + energy_ISP
                    #accelerator baseline
                    if (accelerator_time[i-1] > ISP_time[i]):
                        accelerator_time[i] = accelerator_time[i-1] + latency_accelerator
                    else:
                        accelerator_time[i] = ISP_time[i] + latency_accelerator
                    total_energy_accelerator = total_energy_accelerator + energy_accelerator
                    accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
                    #small npu
                    if (small_flag == 1):
                        smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                        small_flag = 2
                        snpu2 = smallnpu[i] - latency_commit
                        snpu1 = snpu1
                    elif (small_flag ==2):
                        smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                        small_flag = 1
                        snpu1 = smallnpu[i] - latency_commit
                        snpu2 = snpu2
                    total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
                    accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i] 

            else:
                if (i-predict_frame_location)==1 or (i-predict_frame_location)==2 or (i-predict_frame_location)==3:
                    total_energy_smallnpu = total_energy_smallnpu + energy_check
                    if (ssim_real[i] < accuracy):
                        start_time[i] = sensing_time[i-1]
                        #sensing
                        sensing_time[i] = start_time[i] + latency_sensing
                        total_energy_accelerator = total_energy_accelerator + energy_sensing
                        total_energy_smallnpu = total_energy_smallnpu + energy_sensing
                        #ISP
                        if (ISP_time[i-1] > sensing_time[i]):
                            ISP_time[i] = ISP_time[i-1] + latency_ISP
                        else:
                            ISP_time[i] = sensing_time[i] + latency_ISP
                        total_energy_accelerator = total_energy_accelerator + energy_ISP
                        total_energy_smallnpu = total_energy_smallnpu + energy_ISP
                        #accelerator baseline
                        if (accelerator_time[i-1] > ISP_time[i]):
                            accelerator_time[i] = accelerator_time[i-1] + latency_accelerator
                        else:
                            accelerator_time[i] = ISP_time[i] + latency_accelerator
                        total_energy_accelerator = total_energy_accelerator + energy_accelerator
                        accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
                        #smallnpu:
                        if (i-predict_frame_location==1):
                            if (ISP_time[i] + latency_check) > predict_time[i]:
                                if (small_flag==1):
                                    if snpu2 >(ISP_time[i] + latency_check):
                                        smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                                        small_flag = 2
                                        snpu1 = snpu1
                                        snpu2 = smallnpu[i] - latency_commit
                                    else:
                                        smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                        small_flag = 2
                                        snpu1 = snpu1
                                        snpu2 = smallnpu[i] - latency_commit
                                elif (small_flag == 2):
                                    if snpu1 >(ISP_time[i] + latency_check):
                                        smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                                        small_flag = 1
                                        snpu2 = snpu2
                                        snpu1 = smallnpu[i] - latency_commit
                                    else:
                                        smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                        small_flag = 1
                                        snpu2 = snpu2
                                        snpu1 = smallnpu[i] - latency_commit                                
                            else:
                                if (small_flag==1):
                                    if snpu2 >predict_time[i]:
                                        smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                                        small_flag = 2
                                        snpu1 = snpu1
                                        snpu2 = smallnpu[i] - latency_commit
                                    else:
                                        smallnpu[i] = predict_time[i] + latency_smallnpu
                                        small_flag = 2
                                        snpu1 = snpu1
                                        snpu2 = smallnpu[i] - latency_commit
                                elif (small_flag == 2):
                                    if snpu1 >predict_time[i]:
                                        smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                                        small_flag = 1
                                        snpu2 = snpu2
                                        snpu1 = smallnpu[i] - latency_commit
                                    else:
                                        smallnpu[i] = predict_time[i] + latency_smallnpu
                                        small_flag = 1
                                        snpu2 = snpu2
                                        snpu1 = smallnpu[i] - latency_commit 
                        else:
                            if (small_flag==1):
                                if (ISP_time[i] + latency_check) > snpu2:
                                    smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                    snpu1 = snpu1
                                    small_flag = 2
                                    snpu2 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = snpu2 + latency_smallnpu
                                    snpu1 = snpu1
                                    small_flag = 2
                                    snpu2 = smallnpu[i] - latency_commit                                
                            elif (small_flag==2):
                                if (ISP_time[i] + latency_check) > snpu1:
                                    smallnpu[i] = ISP_time[i] + latency_check + latency_smallnpu
                                    snpu2 = snpu2
                                    small_flag = 2
                                    snpu1 = smallnpu[i] - latency_commit
                                else:
                                    smallnpu[i] = snpu1 + latency_smallnpu
                                    snpu2 = snpu2
                                    small_flag = 2
                                    snpu1 = smallnpu[i] - latency_commit         
                        total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
                        accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]

                    else:
                        start_time[i] = sensing_time[i-1]
                        #sensing
                        sensing_time[i] = start_time[i] + latency_sensing
                        total_energy_accelerator = total_energy_accelerator + energy_sensing
                        total_energy_smallnpu = total_energy_smallnpu + energy_sensing
                        #ISP
                        if (ISP_time[i-1] > sensing_time[i]):
                            ISP_time[i] = ISP_time[i-1] + latency_ISP
                        else:
                            ISP_time[i] = sensing_time[i] + latency_ISP
                        total_energy_accelerator = total_energy_accelerator + energy_ISP
                        total_energy_smallnpu = total_energy_smallnpu + energy_ISP
                        #accelerator baseline
                        if (accelerator_time[i-1] > ISP_time[i]):
                            accelerator_time[i] = accelerator_time[i-1] + latency_accelerator
                        else:
                            accelerator_time[i] = ISP_time[i] + latency_accelerator
                        total_energy_accelerator = total_energy_accelerator + energy_accelerator
                        accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
                        #small npu
                        if (small_flag == 1):
                            smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                            small_flag = 2
                            snpu2 = smallnpu[i] - latency_commit
                            snpu1 = snpu1
                        elif (small_flag ==2):
                            smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                            small_flag = 1
                            snpu1 = smallnpu[i] - latency_commit
                            snpu2 = snpu2
                        total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
                        accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]                   

                else:
                    start_time[i] = sensing_time[i-1]
                    #sensing
                    sensing_time[i] = start_time[i] + latency_sensing
                    total_energy_accelerator = total_energy_accelerator + energy_sensing
                    total_energy_smallnpu = total_energy_smallnpu + energy_sensing
                    #ISP
                    if (ISP_time[i-1] > sensing_time[i]):
                        ISP_time[i] = ISP_time[i-1] + latency_ISP
                    else:
                        ISP_time[i] = sensing_time[i] + latency_ISP
                    total_energy_accelerator = total_energy_accelerator + energy_ISP
                    total_energy_smallnpu = total_energy_smallnpu + energy_ISP
                    #accelerator baseline
                    if (accelerator_time[i-1] > ISP_time[i]):
                        accelerator_time[i] = accelerator_time[i-1] + latency_accelerator
                    else:
                        accelerator_time[i] = ISP_time[i] + latency_accelerator
                    total_energy_accelerator = total_energy_accelerator + energy_accelerator
                    accumulation_accelerator = accumulation_accelerator + accelerator_time[i] - start_time[i]
                    #small npu
                    if (small_flag == 1):
                        smallnpu[i] = snpu2 + latency_smallnpu + latency_commit
                        small_flag = 2
                        snpu2 = smallnpu[i] - latency_commit
                        snpu1 = snpu1
                    elif (small_flag ==2):
                        smallnpu[i] = snpu1 + latency_smallnpu + latency_commit
                        small_flag = 1
                        snpu1 = smallnpu[i] - latency_commit
                        snpu2 = snpu2
                    total_energy_smallnpu = total_energy_smallnpu + energy_smallnpu
                    accumulation_smallnpu = accumulation_smallnpu + smallnpu[i] - start_time[i]


    print('acc_base',accumulation_accelerator/frame)    
    print('small',accumulation_smallnpu/frame)      
    print('acc_eng',total_energy_accelerator/frame)
    print('small_eng',total_energy_smallnpu/frame)    


if __name__ == '__main__':
    main()      



                










        	

