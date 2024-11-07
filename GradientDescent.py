import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

import IDSP as idsp

from scipy.sparse import bsr_array

from scipy.optimize import curve_fit

import scipy.constants as const



## physics constants
lamda0 = 480.60202 # FOR our doppler system, we use the 480.60202nm line. Reference NIST

## data extraction parameters
horizontal_span = 100 # the pixels to extract on the ICCD
linear_dispersion = 0.00421 # the linear dispersion, average value 

resolution = 60 # this is the velocity space resolution

center_px = 50 # center pixel is where we find the real 480.602nm is. This is the initial guess

vx = np.linspace(-1e5, 1e5, resolution) # the velocity in one direction
vy = np.linspace(-1e5, 1e5, resolution)

Vx, Vy = np.meshgrid(vx, vy) # the velocity space mesh grid





def get_W(date, shot_in_the_date, pos_i):
    '''
    this function returns the W matrix in IDSP.
    it takes no input...
    as usually the projection matrix is the same.
    the only thing need to take care is the resolution.
    but I will leave it as granted for now.

    this function already takes care of the image drift
    '''

    vx_vy = np.vstack((Vx.ravel(), Vy.ravel())) # the coordinates of all points in the velocity space meshgrid

    v_0 = np.array([[1, 0]]) # the unit vector along 0 degree 
    v_30 = np.array([[np.sqrt(3), 1]]) / 2 # the unit vector along 30 degree
    v_150 = np.array([[-np.sqrt(3), 1]]) / 2 # the unit vector along 150 degree

    projection_0 = v_0 @ vx_vy # this line calculates the velocity along the 30 degree of each point
    projection_30 = v_30 @ vx_vy
    projection_150 = v_150 @ vx_vy

    delta_lamda_0 = projection_0 / const.c * lamda0
    delta_lamda_30 = projection_30 / const.c * lamda0
    delta_lamda_150 = projection_150 / const.c * lamda0
    
    lamda_p_0 = lamda0 - delta_lamda_0
    lamda_p_30 = lamda0 - delta_lamda_30
    lamda_p_150 = lamda0 - delta_lamda_150

    # because there is a bias drift in the ICCD image, we need to correct the center_px
    # to find the real 480.602nm line on the ICCD image
    center_px_correct = center_pxl(date, shot_in_the_date, pos_i)

    wavelength = np.arange(-center_px_correct, 100-center_px_correct) * linear_dispersion + lamda0 

    W_0_idx = np.abs((lamda_p_0 - wavelength[:, None])).argmin(axis=0)
    W_30_idx = np.abs((lamda_p_30 - wavelength[:, None])).argmin(axis=0) + horizontal_span
    W_150_idx = np.abs((lamda_p_150 - wavelength[:, None])).argmin(axis=0) + horizontal_span * 2

    W_idx = np.concatenate([W_0_idx, W_30_idx, W_150_idx])

    column_idx = np.tile(np.arange(0, resolution**2), 3)

    W = bsr_array((np.ones(3 * resolution ** 2), (W_idx, column_idx)), shape=(horizontal_span*3, resolution**2))

    return W





def gaussian(lamda, a0, lamda0, sigma, offset):
    return a0 * np.exp(-0.5 * ((lamda -lamda0)/sigma) ** 2) + offset





def center_pxl(date, shot_in_the_date, pos_i):
    '''
    this function returns the real pixel location of the 480.602nm line.
    it is calculated by taking the mean value of the centers of the line profile of 0deg and 180deg.
    '''
    df0 = idsp.IDSPspectrum(date, shot_in_the_date, pos_i, 0)
    df180 = idsp.IDSPspectrum(date, shot_in_the_date, pos_i, 180)

    [a0, lamda_0, sigma, offset0], pov = curve_fit(gaussian, np.arange(100), df0.intensity, 
                                                p0=[np.max(df0.intensity), center_px, 10, np.mean(df0.intensity[:10])])

    [a180, lamda_180, sigma, offset180], pov = curve_fit(gaussian, np.arange(100), df180.intensity, 
                                                p0=[np.max(df180.intensity), center_px, 10, np.mean(df180.intensity[:10])])

    return np.round((lamda_0 + lamda_180)/2)





def get_P(date, shot, pos_i):
    '''
    this function takes date, shot and position indicator as input,
    and returns the P matrix in IDSP.
    '''
    P = np.zeros(3 * horizontal_span)

    for i, deg in enumerate([0, 30, 150]):
    
        df = idsp.IDSPspectrum(date, shot, pos_i, deg)
        
        [a0, lamda0, sigma, offset], pov = curve_fit(gaussian, np.arange(100), df.intensity, 
                                                     p0=[np.max(df.intensity), center_px, 10, np.mean(df.intensity[:10])])
    
        spec = df.intensity.values - offset
        
        P[i*horizontal_span:(i+1)*horizontal_span] = spec / spec.sum()

    return P





def SIRT(date, shot, pos_i, rho=1e-4, it=int(6e4)):
    '''
    this function takes date and shot as input, pi is the position indicator
    then return the reconstructed field.
    it will automatically call the function to construct P and W for you.

    SIRT stands for simultaneous iteration reconstruction technique, but I just prefer to call it gradient descent.
    '''
    f1 = np.zeros(resolution**2)

    W = get_W(date, shot, pos_i)
    P = get_P(date, shot, pos_i)

    for i in range(it):
        f1 = f1 + rho * W.T @ (P - W @ f1)
        f1[f1<0] = 0

    return f1





def counter_drift(date, shot, pos_i):
    '''
    this function plots the spetrum obtained by the 0deg and 180deg fibers,
    along with the original line, drifted lines, just for visual check.
    '''

    df0 = idsp.IDSPspectrum(date, shot, pos_i, 0)
    df180 = idsp.IDSPspectrum(date, shot, pos_i, 180)
    
    
    plt.figure(figsize=[6, 4])
    
    plt.plot(df0['wavelength'], df0['intensity'])
    plt.plot(df180['wavelength'], df180['intensity'])
    plt.xlabel('Wavelength[nm]')
    plt.ylabel('Intensity[-]')
    
    plt.ylim(bottom=0)
    
    plt.plot([lamda0, lamda0], [0, 175000], 'k--', label='Original %.2fnm' %lamda0)
    # plt.text(lamda0+0.005, 1e5, '$%fnm$' %lamda0)
    
    
    
    [a0, lamda_0, sigma, offset0], pov = curve_fit(gaussian, np.arange(100), df0.intensity, 
                                                    p0=[np.max(df0.intensity), center_px, 10, np.mean(df0.intensity[:10])])
    
    plt.plot(df0.wavelength, gaussian(np.arange(100), a0, lamda_0, sigma, offset0))
    
    
    [a180, lamda_180, sigma, offset180], pov = curve_fit(gaussian, np.arange(100), df180.intensity, 
                                                    p0=[np.max(df180.intensity), center_px, 10, np.mean(df180.intensity[:10])])
    plt.plot(df180.wavelength, gaussian(np.arange(100), a180, lamda_180, sigma, offset180))
    
    
    plt.plot([df0.wavelength[np.round(lamda_0)], df0.wavelength[np.round(lamda_0)]], 
             [offset180, np.max(df0.intensity)],
             'b--', label='center of 0deg')
    plt.plot([df180.wavelength[np.round(lamda_180)], df180.wavelength[np.round(lamda_180)]], 
             [offset180, np.max(df180.intensity)], 
             'r--', label='center of 180deg')
    
    c = (df180.wavelength[np.round(lamda_180)] + df0.wavelength[np.round(lamda_0)]) / 2
    plt.plot([c, c], [0, 175000], ':', label='real %.2fnm' %lamda0)
    
    plt.legend(edgecolor='k', fancybox=False)

    plt.show()





def vz_fitting(f1, plot=False):
    '''
    this function fits the velocity distribution,
    and returns the velocity along the z direction

    f1 should be recovered to 2 dimensional before feeding here,
    but if not, the function will automatically recover it for you(not recommended).
    '''
    if f1.ndim == 1:
        f1 = f1.reshape([resolution, resolution])

    # integrating along the r direction
    vz_dis = f1.sum(axis=0) * np.diff(vy).mean()

    # gaussian fitting the vz profile
    [az, vz_0, sigma_z, offsetz], pov = curve_fit(gaussian, vx, vz_dis, 
                                            p0=[np.max(vz_dis), 0, 2000, np.mean(vz_dis[:10])])

    
    if plot:
        plt.figure(figsize=[4, 3])

        plt.plot(vx, vz_dis)
        plt.xlabel('$v_z(m/s)$')
        plt.ylabel('density(-)')

        plt.plot(vx, gaussian(vx, az, vz_0, sigma_z, offsetz))

        plt.plot([vz_0, vz_0], [0, np.max(vz_dis)], 'k--')
        
        plt.text(0.95, 0.9, '$V_z=%.2fm/s$' %(vz_0), transform=plt.gca().transAxes, ha='right')

        plt.show()

    
    return vz_0





def vr_fitting(f1, plot=False):
    '''
    this function fits the velocity distribution along the z direction,
    and returns the mean velocity

    f1 should be recovered to 2 dimensional before feeding here,
    but if not, the function will automatically recover it for you(not recommended).
    '''
    if f1.ndim == 1:
        f1 = f1.reshape([resolution, resolution])

    # integrating along the r direction
    vr_dis = f1.sum(axis=1) * np.diff(vx).mean()

    # gaussian fitting the vz profile
    [ar, vr_0, sigma_r, offsetr], pov = curve_fit(gaussian, vy, vr_dis, 
                                            p0=[np.max(vr_dis), 0, 2000, np.mean(vr_dis[:10])])


    if plot:
        
        plt.figure(figsize=[4, 3])
        
        plt.plot(vy, vr_dis)
        plt.xlabel('$v_r(m/s)$')
        plt.ylabel('density(-)')

        plt.plot(vy, gaussian(vy, ar, vr_0, sigma_r, offsetr))
        
        plt.plot([vr_0, vr_0], [0, np.max(vr_dis)], 'k--')
        
        plt.text(0.95, 0.9, '$V_r=%.2fm/s$' %(vr_0), transform=plt.gca().transAxes, ha='right')

        plt.show()

    
    return vr_0





def v_dis_plot(f1, date, shot_in_the_date, pos_i, show=True, save=True):
    '''
    this function plots the velocity distrubution.
    just for visual check up.

    the input f1 should be recovered to 2D before feeding here.
    but if it is not, the function will do that for you(not recommended).
    '''
    if f1.ndim == 1:
        f1 = f1.reshape([resolution, resolution])

    
    fig = plt.figure(figsize=[4, 3])
    
    plt.contourf(Vx, Vy, f1, 20)
    plt.colorbar(shrink=0.7)
    
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%i' %(x/1000)))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '%i' %(x/1000)))
    
    plt.xlabel('$V_z[km/s]$')
    plt.ylabel('$V_r[km/s]$')
    
    plt.title('%s-shot%s R=%.3fm' %(date, shot_in_the_date, idsp.IDSP_r(pos_i)))
    
    plt.gca().set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    a=0.6
    ms=1
    ls=1.5
    
    plt.plot(vx[::2], vx[::2]/np.sqrt(3), 'r:', alpha=a, ms=ms, lw=ls)
    plt.plot(vx[::2], vx[::2]/-np.sqrt(3), 'r:', alpha=a, ms=ms, lw=ls)
    plt.plot(vx[::2], vx[::2]*0, 'r:', alpha=a, ms=ms, lw=ls)

    if show:
        plt.show()

    if save:
        fig.savefig('./240906-IDSP result conclusion/%s-shot%s-R=%.3fm.png' %(date, shot_in_the_date, idsp.IDSP_r(pos_i)), dpi=600)