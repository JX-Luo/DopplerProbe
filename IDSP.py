import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob

import sif_parser





def file_path(date, shot_in_the_date):
    '''
    this function returns the sif file path.
    '''
    return glob.glob('/mnt/nifs/%s/shot%s*sif' %(date, shot_in_the_date))[0]





def ICCD_image(date, shot_in_the_date):
    '''
    this function returns the ICCD_image as a 2D array.
    '''
    data, _ = sif_parser.np_open(file_path(date, shot_in_the_date))
    return np.squeeze(data)





def shot_info(date, shot_in_the_date):
    '''
    this function only take date and shot number as input
    and returns the gain, gate width, and gate delay of the shot
    directly from the original sif file.
    this makes sure the ICCD camera parameters are correct, and avoids the error made by human hand recording.

    Parameters
    ----------
    date: the date of experiment
    shot_in_the_date: the shot number within the date

    Returns
    -------
    gate_delay: the gate delay as recorded by the camera, the unit is [s] 
    gate_width: the gate width as recorded in the camera, the unit is [s]
    gain
    '''
    
    # file_path = glob.glob('/mnt/nifs/%s/shot%s*sif' %(date, shot_in_the_date))
    
    _, info = sif_parser.np_open(file_path(date, shot_in_the_date))

    gate_delay = int(info['GateDelay'] * 1e6)
    gate_width = info['GateWidth'] * 1e6
    gain = info['GateGain']

    return gate_delay, gate_width, gain





def spectrum_path_finder(date, shot_in_the_date, position_indicator, angle):
    '''
    this function takes only date, shot in the date, 
    position indicator(int from 1 to 7), and angle(0, 30, 150 or 180) as input
    and returns the spectrum csv file's path

    Parameters
    ----------
    date: date
    shot_in_the_date: shot in the date
    position_indicator: int, from 1 to 7
    angle: 0, 30, 150, or 180
    
    Returns
    -------
    filepath: the file path of the stored spectrum
    '''
    
    if (angle == 150) | (angle==180):
        file_path = glob.glob('./%s/shot%s*/formatted_data/Group%s_%s-%s*.csv' 
                          %(date, shot_in_the_date, 8-position_indicator, position_indicator+7, 180-angle))
    else:
        file_path = glob.glob('./%s/shot%s*/formatted_data/Group%s_%s-%s*.csv' 
                          %(date, shot_in_the_date, 8-position_indicator, position_indicator, angle))
    
    
    return file_path[0]





def IDSPspectrum(date, shot_in_the_date, position_indicator, angle):
    '''
    this function returns the spectrum dataframe
    '''
    file_path = spectrum_path_finder(date, shot_in_the_date, position_indicator, angle)
    file = pd.read_csv(file_path)
    
    return file





def IDSP_r(position_indicator):
    '''
    this function takes only the position indicator as input, 
    and returns the position of measurement in [m]
    '''
    return 9e-2 + position_indicator*25e-3