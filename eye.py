import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
import math
import time

mplstyle.use('fast')

def eye_plot(y, T, acc, eye_num=1):
    # Save certain information
    volt_per_dot = (max(y)-min(y))/(acc-1)
    time_per_dot = T/acc
    
    # Normalize data
    y = (acc-1) * (y-min(y)) / (max(y)-min(y))
    y = y.astype(int)
    y[y > acc-1] = acc-1
    y[y < 0] = 0
    
    # The first data to plot
    trigger = np.argwhere(y == acc//2).T[0]
    trigger = np.mod(trigger, acc)
    
    if len(trigger) != 0:
        if max(trigger)-min(trigger) > 0.9*acc:
            for i in range(len(trigger)):
                if trigger[i] < 0.5*acc:
                    trigger[i] = trigger[i]+acc
        location = int(np.mean(trigger))
        
    else:
        greater = y > acc//2
        for n in range(acc+acc//2+1, len(greater)):
            if greater[n] != greater[acc+acc//2]:
                location = n
                break
    
    location += acc//2
    
    t_eye = np.array([], dtype=int)
    y_eye = np.array([], dtype=int)
    for n in range(eye_num+1):
        period = (len(y)-location) // (acc*(eye_num+1))
        t_eye = np.append(t_eye, np.tile(np.array(range((eye_num+1)*acc)), period))
        y_eye = np.append(y_eye, y[location: location+period*(eye_num+1)*acc])
        
        location += acc
    # Create orientation
    [orientation, times] = np.unique(np.vstack((t_eye, y_eye)).T, axis=0, return_counts=True)

    # Create image
    canvas = np.zeros((acc, acc*(eye_num+1)), dtype=int)
    canvas[orientation[:, 1], orientation[:, 0]] = times

    # Calculate eye amplitude and eye height
    eye_amp = np.array([])
    eye_height = np.array([])
    for n in range(eye_num):
        # Get sample
        sample = np.sum(canvas[:, int(0.9*acc + n*acc): int(1.1*acc + n*acc)], axis=1)
        # Eye amplitude
        mean_bt = np.sum(np.multiply(np.array(range(acc//2)), sample[: acc//2])) / \
                  np.sum(sample[: acc//2])
        mean_tp = np.sum(np.multiply(np.array(range(acc//2, acc)), sample[acc//2:])) / \
                  np.sum(sample[acc//2:])
        eye_amp = np.append(eye_amp, (mean_tp - mean_bt) * volt_per_dot)
        
        # Eye height
        dist2_bt = np.power(np.array(range(acc//2)) - mean_bt, 2)
        dist2_tp = np.power(np.array(range(acc//2, acc)) - mean_tp, 2)
        var_bt = np.power(np.sum(np.multiply(dist2_bt, sample[: acc//2])) /
                          np.sum(sample[: acc//2]), 0.5)
        var_tp = np.power(np.sum(np.multiply(dist2_tp, sample[acc//2:])) /
                          np.sum(sample[acc//2:]), 0.5)
        eye_height = np.append(eye_height, (mean_tp-mean_bt)* volt_per_dot - 3*(var_tp+var_bt) * volt_per_dot)
        
    eye_amp = np.mean(eye_amp)
    eye_height = np.mean(eye_height)

    # Calculate eye width
    location_jitter = np.array([])
    for n in range(eye_num+1):
        # Get sample
        sample = canvas[int(mean_bt): int(mean_tp)+1, n*acc: (n+1)*acc]
        centre = np.sum(np.multiply(sample, np.array(range(acc))), axis=1) / \
                 np.sum(sample, axis=1)
        dist2 = np.power(np.tile(np.array(range(acc)), len(centre)).reshape(-1, acc) -
                         np.repeat(centre, acc).reshape(-1, acc), 2)
        var = np.power(np.sum(np.multiply(sample, dist2), axis=1) /
                       np.sum(sample, axis=1), 0.5)
        location_jitter = np.append(location_jitter, [centre[np.nanargmin(var)], np.nanmin(var)])
    location_jitter = location_jitter.reshape(-1, 2)
    eye_width = (eye_num*acc + (location_jitter[-1, 0]-location_jitter[0, 0]) - \
                3 * (location_jitter[-1, 1]+location_jitter[0, 1]) - \
                6 * np.sum(location_jitter[1:-1, 1])) / acc * T / eye_num

    return eye_amp, eye_height, eye_width
