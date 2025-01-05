import numpy as np
import re
import os
import random
from load_ann_noise import inference
import load_ann_ideal
   

def tx_si(in_real, period, step, eq_state, inpvh, cload, rxr, rxv, trf, tx_name):
    npy_len = in_real.shape[0]
    idealdata = np.zeros((1, npy_len + 3))
    idealdata[0, 0:npy_len] = in_real
    
    if tx_name == 'b1':
        idealdata[0, -3] = cload/20.0
        idealdata[0, -2] = rxr
        idealdata[0, -1] = rxv
    
        if eq_state == True:
            fir = 1 - in_real[0]/np.max(in_real)
            out_ideal = load_ann_ideal.inference('b1_eq', npy_len, 5, 45, np.append(idealdata, fir).reshape((1, npy_len + 4)))
        else:
            out_ideal = load_ann_ideal.inference_noeq('b1_ideal', npy_len, 5, 45, idealdata)
    
    elif tx_name == 'b2':
        idealdata[0, -3] = cload
        idealdata[0, -2] = rxr
        idealdata[0, -1] = rxv
    
        if eq_state == True:
            fir = 1 - in_real[0]/np.max(in_real)
            out_ideal = load_ann_ideal.inference('b2_eq', npy_len, 1, 45, np.append(idealdata, fir).reshape((1, npy_len + 4)))
        else:
            out_ideal = load_ann_ideal.inference_noeq('b2_ideal', npy_len, 1, 45, idealdata)   
        
    return out_ideal

def tx_pi(inpvh, cload, rxr, rxv, noise, in_real, tx_name, vcc_real = []):
    noise = np.array(noise)
    if vcc_real == []:
        npy_len = len(noise)
        vcc_real = inpvh - noise
    else:
        npy_len = len(vcc_real)
        vcc_real = np.array(vcc_real)
    
    testdata = np.zeros((2, npy_len + 3))
    testdata[0, 0:npy_len] = vcc_real
    testdata[1, 0:len(in_real)] = in_real
    if tx_name == 'b1':
        testdata[0, -3] = cload/20.0
        testdata[0, -2] = rxr
        testdata[0, -1] = rxv
    
        add_noise = inference('b1_noise', npy_len, 5, 45, testdata)
    
    elif tx_name == 'b2':
        testdata[0, -3] = cload
        testdata[0, -2] = rxr
        testdata[0, -1] = rxv
    
        add_noise = inference('b2_noise', npy_len, 1, 45, testdata)
    
    return add_noise
