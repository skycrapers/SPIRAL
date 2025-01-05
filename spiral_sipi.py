import numpy as np
import math
import re
import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skrf as rf       # pip install scikit-rf
from math import pi
import time
from scipy.interpolate import interp1d
from scipy.fftpack import ifft
from math import log2, ceil, pi
import logging
from _brescount import bres_curve_count as _bres_curve_count
from si_pi_pre import tx_si, tx_pi
import eye
    
def to_float(s):
    if s[-1]=='m':
        res = float(s[0:-1])*1e-3
    elif s[-1]=='u':
        res = float(s[0:-1])*1e-6 
    elif s[-1]=='n':
        res = float(s[0:-1])*1e-9  
    elif s[-1]=='p':
        res = float(s[0:-1])*1e-12
    elif s[-1]=='f':
        res = float(s[0:-1])*1e-15
    elif s[-1]=='a':
        res = float(s[0:-1])*1e-18
    elif s[-1]=='k':
        res = float(s[0:-1])*1e3
    else:
        res = float(s)
    
    return res   

def square2npy(inpvl,inpvh,tb,tr,tf,th,period,simtime,step):
    tb=tb*2
    tr=tr*2
    tf=tf*2
    th=th*2
    period=period*2
    simtime=simtime*2
    tb_num1 = int(tb/step+0.5)
    tr_seq = np.arange(inpvl, inpvh-(inpvh-inpvl)/(tr/step)/2, (inpvh-inpvl)/(tr/step))
    th_num = int(th/step+0.5)
    tf_seq = np.arange(inpvh, inpvl-(inpvl-inpvh)/(tf/step)/2, (inpvl-inpvh)/(tf/step))
    tb_num2 = int((period-tb-tr-th-tf)/step+0.5)
    inseq = [inpvl]*tb_num1+tr_seq.tolist()+[inpvh]*th_num+tf_seq.tolist()+[inpvl]*tb_num2  # one period
    per_num = int(simtime//period)
    rest_num = int((simtime%period)/step+0.5)
    inseq = inseq*per_num+inseq[0:rest_num+1]
    innpy = np.array(inseq[::2])
    
    return innpy
    
def bi2npy(bi_seq,inpv,trf,period,step):
    npyres = np.zeros(int(len(bi_seq)*period/step+0.5)+int(period/step+0.5)+1,)
    for i in range(len(bi_seq)):
        if bi_seq[i] == '0':
            uni = np.array([0]*(int(period/step+0.5)+int(period/step+0.5)+1))
        else:
            uni = square2npy(0,inpv,(period-trf)/2.0,trf,trf,period-trf,2*period,2*period,step)
        npyres[int(i*period/step+0.5):int((i+2)*period/step+0.5)+1] = npyres[int(i*period/step+0.5):int((i+2)*period/step+0.5)+1]+uni   
    
    return npyres               

def bi2pwl(bi_seq,inpv,trf,period,step):
    npyres = bi2npy(bi_seq,inpv,trf,period,step)
    pwls = ''    
    for i in range(npyres.shape[0]):
        if i == 0 or i == npyres.shape[0]-1:
            pwls = pwls+str(i*step)+'ns '+str(npyres[i])+' '
        else:
            if npyres[i] == npyres[i-1] and npyres[i] == npyres[i+1]:
                continue
            else:
                pwls = pwls+str(i*step)+'ns '+str(npyres[i])+' '
    
    return pwls

def npy2pwl(innpy,step):
    pwls = ''
    for i in range(innpy.shape[0]):
        if i == 0 or i == innpy.shape[0]-1:
            pwls = pwls+str(i*step)+'ns '+str(innpy[i])+' '
        else:
            if innpy[i] == innpy[i-1] and innpy[i] == innpy[i+1]:
                continue
            elif innpy[i]-innpy[i-1] == innpy[i+1]-innpy[i]:
                continue
            else:
                pwls = pwls+str(i*step)+'ns '+str(innpy[i])+' '
        
    return pwls    

def conv_d(x,h):
    xlen = len(x)
    hlen = len(h)
    xnpy = np.zeros((xlen+hlen-1, hlen))
    xz = [0]*(hlen-1)+x+[0]*(hlen-1)
    hz = np.array(list(reversed(h)))
    for i in range(xlen+hlen-1):
        xnpy[i,:] = np.array(xz[i:i+hlen])
    ylist = np.dot(xnpy,hz).tolist()
        
    return ylist       

def equalizer_(bi_seq, inpv, trf, fir_tap, ul, step):
    pwl1 = bi2npy('1',inpv,trf,ul,step)
    ui_sample = int(ul/step+0.5)
    tap_num = len(fir_tap)
    fir_inter = np.zeros((tap_num, ui_sample))
    fir_inter[:,0] = np.array(fir_tap)
    fir_inter = fir_inter.reshape(tap_num*ui_sample, )
    eq1 = np.array(conv_d(pwl1.tolist(),fir_inter.tolist()))[0:3*ui_sample]
    offset = eq1[0]
    eq1 = eq1 - offset
    add_res = np.zeros(((len(bi_seq)+2)*ui_sample,))
    for i in range(len(bi_seq)):
        if bi_seq[i] == '0':
            pass
        else:
            add_res[i*ui_sample:(i+3)*ui_sample] = add_res[i*ui_sample:(i+3)*ui_sample]+eq1
    add_res = add_res+offset-np.min(eq1)

    return add_res  

def s2tf(ntwk, inport, outport):
    sparam = ntwk.s
    freq_samp = len(sparam)
    sp = np.zeros((freq_samp, 2, 2), dtype=complex)
    sp[:, 0, 0] = sparam[:, inport, inport]
    sp[:, 0, 1] = sparam[:, inport, outport]
    sp[:, 1, 0] = sparam[:, outport, inport]
    sp[:, 1, 1] = sparam[:, outport, outport]

    gamma_l = 1
    zs = 0
    zo = 50
    gamma_s = (zs - zo) / (zs + zo)
    gamma_in = sp[:, 0, 0] + (sp[:, 0, 1] * sp[:, 1, 0] * gamma_l / (1 - sp[:, 1, 1] * gamma_l))

    tf = (sp[:, 1, 0] * (1 + gamma_l) * (1 - gamma_s)) / (2 * (1 - sp[:, 1, 1] * gamma_l) * (1 - gamma_in * gamma_s))

    return tf

def get_impulse(f, tf, dt, T):
    n_req = round(T / dt)
    logging.debug('Number of time points requested: {}'.format(n_req))

    n = 1 << int(ceil(log2(n_req)))   
    logging.debug('Number of IFFT points: {}'.format(n))

    df = 1 / (n * dt)

    f = f.copy()
    tf = tf.copy()

    if f[0] == 0:
        logging.debug('Removing imaginary part of tf[0]')

        assert is_mostly_real(tf[0])
        tf[0] = tf[0].real

    ma = np.abs(tf)
    ph = np.unwrap(np.angle(tf))

    # add DC component if necessary
    if f[0] != 0:
        logging.debug('Adding point f[0]=0, tf[0]=abs(tf[1])')

        f = np.concatenate(([0], f))
        ma = np.concatenate(([ma[0]], ma))
        ph = np.concatenate(([0], ph))

    logging.debug('Interpolating magnitude and phase.')
    f_interp = np.arange(n / 2) * df
    ma_interp = interp1d(f, ma, bounds_error=False, fill_value=(ma[0], 0))(f_interp)
    ph_interp = interp1d(f, ph, bounds_error=False, fill_value=(0, 0))(f_interp)

    # create frequency response vector needed for IFFT
    logging.debug('Creating the frequency response vector.')
    Gtilde = np.zeros(n, dtype=np.complex128)
    Gtilde[:(n // 2)] = ma_interp * np.exp(1j * ph_interp)
    Gtilde[((n // 2) + 1):] = np.conjugate(Gtilde[((n // 2) - 1):0:-1])

    # compute impulse response
    y_imp = n * df * (ifft(Gtilde)[:int(n_req)])

    if not is_mostly_real(y_imp):
        raise Exception('IFFT contains unacceptable imaginary component.')
    y_imp = np.real(y_imp)

    return np.arange(n_req) * dt, y_imp

def is_mostly_real(v, ratio=1e-6):
    return np.all(np.abs(np.imag(v) / np.real(v)) < ratio)

def s2hn(spara_file, i_port, o_port, dt, hn_samples=1000, vtr_fit=True):
    ntwk = rf.Network(spara_file)
    tf = s2tf(ntwk, i_port, o_port)
    if vtr_fit:
        ntwk_new = rf.Network(frequency=ntwk.frequency, s=tf)
        vf1 = rf.VectorFitting(ntwk_new)
        vf1.max_iterations = 2000
        vf1.vector_fit(n_poles_real=70, n_poles_cmplx=80, parameter_type='s')
        tf_fit = vf1.get_model_response(0, 0, np.linspace(0, ntwk_new.f[-1], hn_samples))
        [t, hn] = get_impulse(np.linspace(0, ntwk_new.f[-1], hn_samples), tf_fit, dt, hn_samples*dt)
    else:
        [t, hn] = get_impulse(ntwk.f, tf, dt, hn_samples*dt)
    hn = hn * dt

    return hn

def vcc_noise_gen(period,simtime,step,vcc):  
    noise_per = period/2.0
    noise_gen = []
    per_num = int(simtime/noise_per)
    for j in range(per_num):
        noise_v = random.uniform(0.05*vcc,0.35*vcc)
        for i in range(int(noise_per/2.0/step+0.5)):        
            noise_gen.append(noise_v/(noise_per/2.0/step)*i)
        for i in range(int(noise_per/2.0/step+0.5)):
            noise_gen.append(noise_v-noise_v/(noise_per/2.0/step)*i)
    
    rest_time = simtime-per_num*noise_per
    noise_gen = noise_gen + noise_gen[0:int(rest_time/step+0.5)+1]
    
    return noise_gen

def unit_save(inpv,cload,trf,result,step,link_num,in_index,out_index, unit_len,tail_len,upath):
    tail = int(tail_len/step+0.5)
    np.save(upath+str(link_num)+'p'+str(in_index)+'i'+str(out_index)+'o_'+str(cload)+'_'+str(inpv)+'_'+str(trf)+'_'+str(unit_len/2)+'_'+str(step)+'ui.npy', np.append(result[0:int(unit_len/step+0.5)+tail],result[-3:]))

def unit_add(cload,inpv,trf,bi_seq,per,step,link_num,in_index,out_index,unit_len,tail_len,upath):
    tail = int(tail_len/step+0.5)   
    outres = np.zeros(int(len(bi_seq)*per/step+per/step+0.5)+tail, dtype = complex) 
    offset = np.load(upath+str(link_num)+'p'+str(in_index)+'i'+str(out_index)+'o_'+str(cload)+'_'+str(inpv)+'_'+str(trf)+'_'+str(unit_len/2)+'_'+str(step)+'ui.npy')[-1]
    unit_1 = np.load(upath+str(link_num)+'p'+str(in_index)+'i'+str(out_index)+'o_'+str(cload)+'_'+str(inpv)+'_'+str(trf)+'_'+str(unit_len/2)+'_'+str(step)+'ui.npy')[0:-1]
    
    for i in range(len(bi_seq)):        
        if bi_seq[i] == '1':
            unit = unit_1            
        else:
            unit = np.zeros(int(unit_len/step+0.5)+tail,)
         
        for j in np.arange(0,per-unit_len/4.0,unit_len/2.0):
            outres[int((i*per+j)/step+0.5):int((i*per+j+unit_len)/step+0.5)+tail] = outres[int((i*per+j)/step+0.5):int((i*per+j+unit_len)/step+0.5)+tail] + unit[0:int(unit_len/step+0.5)+tail]
        
    return outres, offset

def eyediagram(y, window_size, offset=0, colorbar=True, **imshowkwargs):
    counts = grid_count(y, window_size, offset)
    counts = counts.astype(np.float32)
    counts[counts == 0] = np.nan
    ymax = y.max()
    ymin = y.min()
    yamp = ymax - ymin
    min_y = ymin - 0.05 * yamp
    max_y = ymax + 0.05 * yamp
    plt.imshow(counts.T[::-1, :],
               extent=[0, 2, min_y, max_y], interpolation='nearest',  cmap = 'jet')
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.grid(color='grey',linestyle=':')
    if colorbar:
        plt.colorbar()

def grid_count(y, window_size, offset=0, size=None, fuzz=True, bounds=None):
    if size is None:
        size = (800, 640)
    height, width = size
    dt = width / window_size
    counts = np.zeros((width, height), dtype=np.int32)

    if bounds is None:
        ymin = y.min()
        ymax = y.max()
        yamp = ymax - ymin
        ymin = ymin - 0.05 * yamp
        ymax = ymax + 0.05 * yamp
    else:
        ymin, ymax = bounds

    start = offset
    while start + window_size < len(y):
        end = start + window_size
        yy = y[start:end + 1]
        k = np.arange(len(yy))
        xx = dt * k
        if fuzz:
            f = interp1d(xx, yy, kind='cubic')
            jiggle = dt * (np.random.beta(a=3, b=3, size=len(xx) - 2) - 0.5)
            xx[1:-1] += jiggle
            yd = f(xx)
        else:
            yd = yy
        iyd = (height * (yd - ymin) / (ymax - ymin)).astype(np.int32)
        _bres_curve_count(xx.astype(np.int32), iyd, counts)

        start = end
    return counts

def main():
    backupfile = 'b1_l3.txt'
    with open(backupfile, 'r') as f:
        bkuplines = f.readlines()
    linkname = re.split('\s+', bkuplines[1])[0]
    txname = re.split('\s+', bkuplines[2])[0]
    sysname = re.split('\s+', bkuplines[3])[0]    
    ul = re.split('\s+', bkuplines[4])[0:-1]
    ul = [float(x) for x in ul]   
    fl = float(re.split('\s+', bkuplines[5])[0])
    fh = float(re.split('\s+', bkuplines[5])[1])
    zero_f = bool(re.split('\s+', bkuplines[5])[2]=='True') 
    stype = re.split('\s+', bkuplines[5])[3]    
    rxr_list = re.split('\s+', bkuplines[6])[0:-1]
    rxr_list = [float(x) for x in rxr_list] 
    rxv_list = re.split('\s+', bkuplines[7])[0:-1]
    rxv_list = [float(x) for x in rxv_list]
    liner_list = re.split('\s+', bkuplines[8])[0:-1]
    liner_list = [float(x) for x in liner_list]
    cload_list = re.split('\s+', bkuplines[9])[0:-1]
    cload_list = [float(x) for x in cload_list]
    trf_list = re.split('\s+', bkuplines[10])[0:-1]
    trf_list = [float(x) for x in trf_list]
    inpvh_list = re.split('\s+', bkuplines[11])[0:-1]
    inpvh_list = [float(x) for x in inpvh_list]        
    bi_list = re.split('\s+', bkuplines[12])[0:-1]
    bi_list = [str(x) for x in bi_list]
    step = float(re.split('\s+', bkuplines[13])[0])
    is_pi = bool(re.split('\s+', bkuplines[14])[0]=='True')
    eq_state = bool(re.split('\s+', bkuplines[15])[0]=='True') 
    if eq_state == True:
        fir_c0 = float(re.split('\s+', bkuplines[15])[1])
        fir_tap = [fir_c0, fir_c0 - 1]
    else:
        fir_tap = [1, 0]
    
    if is_pi == True:
        pi_type = 'with_noise'
    else:
        pi_type = 'ideal'
    
    if txname == 'b1':
        tail = 20
    elif txname == 'b2':
        tail = 0.5
    unit_bi_len = math.ceil(tail/min(ul))+1 
    
    period_list = ul
    th_list = np.divide(period_list,2.0)-trf_list
    link_num = len(bi_list)
    
    if stype == 'dec':
        spara_name = './link_rx/'+linkname+'/lr'+str(int(fl))+'_'+str(int(fh))+stype
    elif stype == 'lin' and zero_f == True:
        spara_name = './link_rx/'+linkname+'/lr0_'+str(int(fl))+'_'+str(int(fl))+stype
    elif stype == 'lin' and zero_f == False:
        spara_name = './link_rx/'+linkname+'/lr'+str(int(fl))+'_'+str(int(fl))+stype
    
    h_sample = 1000  
    threshold = 0.0
    n_mat = np.ones((link_num, link_num), dtype=int)
    hn_npy_name = spara_name+'_th'+str(threshold)+'_step'+str(step)+'_new.npy'

    if not os.path.exists(hn_npy_name):
        hn_npy = np.zeros((h_sample+1,link_num,link_num))
        for i in range(link_num):  # i: output
            for j in range(link_num):  # j: input
                if n_mat[i, j]:
                    print(i, j)
                    hn_npy[0:h_sample, i, j] = np.array(s2hn(spara_name +'_'+str(2*j)+'_'+str(2*i+1) + '.s2p', 0, 1, step*1e-9, hn_samples = h_sample, vtr_fit=False)).reshape(h_sample, )
                    hn_npy[h_sample,i,j] = n_mat[i,j]
        np.save(hn_npy_name,hn_npy)
        hn = hn_npy[0:h_sample,:,:]
    else:
        hn_npy = np.load(hn_npy_name)
        hn = hn_npy[0:h_sample,:,:]
        n_mat = hn_npy[h_sample,:,:]
    
    global noise_list
    noise_list = []
    if pi_type == 'with_noise':
        for i in range(link_num):
            if txname == 'b1':
                noise_list.append(vcc_noise_gen(random.randint(1,6),(len(bi_list[i])+1)*max(ul)+tail,step,inpvh_list[i]))
                
            elif txname == 'b2':
                noise_list.append(vcc_noise_gen(random.choice([0.02,0.03,0.04,0.05,0.06,0.07,0.08]),(len(bi_list[i])+1)*max(ul)+tail,step,inpvh_list[i]))
    
    # save pulse response
    if not os.path.exists('./unit/'+txname+'_'+linkname+'/'):
        os.makedirs('./unit/'+txname+'_'+linkname+'/')
    for in_index in range(link_num):
        if eq_state == True:
            if os.path.exists('./unit/'+txname+'_'+linkname+'/eq'+str(fir_tap[0])+'_'+str(link_num)+'p'+str(in_index*2)+'i'+str(in_index*2+1)+'o_'+str(cload_list[in_index])+'_'+str(inpvh_list[in_index])+'_'+str(trf_list[in_index])+'_'+str(ul[in_index]*2/2)+'_'+str(step)+'ui.npy'):
                continue
        else:
            if os.path.exists('./unit/'+txname+'_'+linkname+'/'+str(link_num)+'p'+str(in_index*2)+'i'+str(in_index*2+1)+'o_'+str(cload_list[in_index])+'_'+str(inpvh_list[in_index])+'_'+str(trf_list[in_index])+'_'+str(ul[in_index]*2/2)+'_'+str(step)+'ui.npy'):
                continue
                
        bis_list = ['0'*unit_bi_len]*link_num
        bis_list[in_index]='1'+'0'*(unit_bi_len-1)

        if eq_state == True:  
            in_eq = equalizer_(bis_list[in_index], inpvh_list[in_index], trf_list[in_index], fir_tap, period_list[in_index], step)
            tx_out = tx_si(in_eq, period_list[in_index], step, eq_state, inpvh_list[in_index], cload_list[in_index], rxr_list[in_index], rxv_list[in_index], trf_list[in_index], txname)
        else:
            tx_in = bi2npy(bis_list[in_index],inpvh_list[in_index],trf_list[in_index],period_list[in_index],step)
            tx_out = tx_si(tx_in, period_list[in_index], step, eq_state, inpvh_list[in_index], cload_list[in_index], rxr_list[in_index], rxv_list[in_index], trf_list[in_index], txname)
        

        for i in range(link_num):
            if n_mat[i, in_index]:
                out = np.array(conv_d((tx_out-tx_out[0]).tolist(), (hn[:,i,in_index]).tolist()))
                save_info = np.append(out, tx_out[0])

                if eq_state == True:
                    unit_save(inpvh_list[in_index],cload_list[in_index],trf_list[in_index],save_info,step,link_num,in_index*2,i*2+1,ul[in_index]*2,tail,'./unit/'+txname+'_'+linkname+'/eq'+str(fir_tap[0])+'_')
                else:
                    unit_save(inpvh_list[in_index],cload_list[in_index],trf_list[in_index],save_info,step,link_num,in_index*2,i*2+1,ul[in_index]*2,tail,'./unit/'+txname+'_'+linkname+'/')
    
    if pi_type == 'with_noise':  
        tx_noise_list = []
        for in_index in range(link_num):
            in_real = bi2npy(bi_list[in_index],inpvh_list[in_index],trf_list[in_index],period_list[in_index],step)
            tx_noise = tx_pi(inpvh_list[in_index], cload_list[in_index], rxr_list[in_index], rxv_list[in_index], noise_list[in_index], in_real, txname)
            tx_noise_list.append(tx_noise)
            
    # pulse response superposition
    add_res = []
    for i in range(link_num):  # i: output
        add_out = 0
        for j in range(link_num):  # j: input
            if n_mat[i, j]:
                if eq_state == True:
                    addone, offset = unit_add(cload_list[j],inpvh_list[j],trf_list[j],bi_list[j],period_list[j],step,link_num,j*2,i*2+1,ul[j]*2,tail,'./unit/'+txname+'_'+linkname+'/eq'+str(fir_tap[0])+'_')
                else:
                    addone, offset = unit_add(cload_list[j],inpvh_list[j],trf_list[j],bi_list[j],period_list[j],step,link_num,j*2,i*2+1,ul[j]*2,tail,'./unit/'+txname+'_'+linkname+'/')
                
                if pi_type == 'with_noise':
                    noise_part = np.array(conv_d(tx_noise_list[j].tolist(), hn[:, i, j].tolist()))[0:len(addone)]
                else:
                    noise_part = 0
                if i==j:
                    off_true = offset
                
                add_out = add_out + addone + noise_part
        offset_final = (rxr_list[i]*2*off_true+liner_list[i]*rxv_list[i]*2)/(rxr_list[i]*2+liner_list[i]*2)
        add_res.append(add_out+offset_final)
    
    for i in range(link_num):
        plt.figure('out'+str(i))
        plt.plot((np.arange(len(add_res[i]))*step), (add_res[i].real))
        plt.xlabel('t(ns)')
        plt.ylabel('Voltage(V)')
        plt.show()
    
    # eyediagram
    bi_list = []
    for j in range(link_num):
        bi_seq = ''
        for i in range(500):
            bi = random.randint(0,1)
            bi_seq = bi_seq+str(bi)
        bi_list.append(bi_seq)
    noise_list_e = []
    tx_noise_list_e = []
    if pi_type == 'with_noise':
        for i in range(link_num):
            noise_total_len = (len(bi_list[i])+1)*ul[i]+tail
            noise_e = []
            for j in range(10):
                if txname == 'b1':
                    noise_e = noise_e + vcc_noise_gen(random.randint(1,6),noise_total_len/10.0,step,inpvh_list[i])
                elif txname == 'b2':
                    noise_e = noise_e + vcc_noise_gen(random.choice([0.02,0.03,0.04,0.05,0.06,0.07,0.08]),noise_total_len/10.0,step,inpvh_list[i])
            noise_list_e.append(noise_e)
    if pi_type == 'with_noise':
        for i in range(link_num):
            in_real = bi2npy(bi_list[i],inpvh_list[i],trf_list[i],period_list[i],step)
            tx_noise_e = tx_pi(inpvh_list[i], cload_list[i], rxr_list[i], rxv_list[i], noise_list_e[i], in_real, txname)
            tx_noise_list_e.append(tx_noise_e)
    add_res_e = []
    for i in range(link_num):  # i: output
        add_out = 0
        for j in range(link_num):  # j: input
            if n_mat[i, j]:
                if eq_state == True:
                    addone, offset = unit_add(cload_list[j],inpvh_list[j],trf_list[j],bi_list[j],period_list[j],step,link_num,j*2,i*2+1,ul[j]*2,tail,'./unit/'+txname+'_'+linkname+'/eq'+str(fir_tap[0])+'_')
                else:
                    addone, offset = unit_add(cload_list[j],inpvh_list[j],trf_list[j],bi_list[j],period_list[j],step,link_num,j*2,i*2+1,ul[j]*2,tail,'./unit/'+txname+'_'+linkname+'/')
                if pi_type == 'with_noise':
                    noise_part_e = np.array(conv_d(tx_noise_list_e[j].tolist(), hn[:, i, j].tolist()))[0:len(addone)]
                else:
                    noise_part_e = 0
                if i==j:
                    off_true = offset
                add_out = add_out + addone + noise_part_e
        
        offset_final = (rxr_list[i]*2*off_true+liner_list[i]*rxv_list[i]*2)/(rxr_list[i]*2+liner_list[i]*2)
        add_res_e.append(add_out+offset_final)
    
    if not os.path.exists('./eyediagram_output'): 
        os.mkdir('./eyediagram_output')
    
    for i in range(link_num):
        plt.figure('eyediagram'+str(i))
        samples_per_symbol = int(ul[i]/step+0.5)
        y=np.array(add_res_e[i].real)
        eyediagram(y, 2*samples_per_symbol, offset=220, cmap=plt.cm.coolwarm)
        parameter = eye.eye_plot(y=y, T=ul[i]*1e-9, acc=samples_per_symbol, eye_num=1)
        plt.savefig('./eyediagram_output/eyediagram'+str(i)+'.jpg')
        print('link'+str(i)+':')
        print('Amplitude:\t%.3fV' % parameter[0])
        print('Height:\t\t%.3fV' % parameter[1])
        print('Width:\t\t%.4gs' % parameter[2])
        print('\n')
    plt.show()
    

if __name__ == '__main__':
    main()
    