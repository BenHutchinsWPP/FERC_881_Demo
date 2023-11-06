#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for calculating line ratings based on the IEEE
738-2012 standard.
"""
from collections import namedtuple
from datetime import timedelta
from typing import Union
import numpy as np
import pandas as pd
def calculate_thermal_rating(
    T_a: Union[np.ndarray, float],
    T_s: Union[np.ndarray, float],
    D: Union[np.ndarray, float],
    emis: Union[np.ndarray, float],
    absorp: Union[np.ndarray, float],
    R: Union[np.ndarray, float],
    W_v: Union[np.ndarray, float],
    W_a: Union[np.ndarray, float],
    elev: Union[np.ndarray, float],
    air_qual: Union[np.ndarray, float],
    year: Union[np.ndarray, int],
    day_of_year: Union[np.ndarray, int],
    start_hour: Union[np.ndarray, int],
    latitude: Union[np.ndarray, float],
    longitude: Union[np.ndarray, float],
    time_zone: Union[np.ndarray, int],
    line_azimuth: Union[np.ndarray, float],
    verbose=0) ->Union[np.ndarray, int]:
    """
    NOTE!:  Function developed for demonstration purposes only.  Not intended
    for use in any real-world application, and has not been sufficiently quality
    controlled for any purpose other than demonstrations.  Provided for use at 
    the user's own risk and without warranty.
    Function to calculate a transmission line thermal rating in terms of
    current (Amps).
    
    Output is a scalar if inputs are all scalars, or a 1D array if any of
    the inputs are a 1D array.
    
    Calculation based on IEEE-738-2012, except for certain
    equations related to determination of solar declination, zenith, and 
    azimuth, which were adapted from NOAA General Solar Position Calculations 
    document, available at https://gml.noaa.gov/grad/solcalc/solareqns.PDF. 
    The NOAA equations were used (1) to facilitate direct calculation of 
    sunrise/sunset times (to identify periods where solar heating = 0), and
    (2) to correct certain calculated parameters to reflect longitude (which
    the IEEE equations do not consider).
    


    This function does not account for other potential line limits, such as
    voltage or stability limits.
    
    Parameters
    ----------
    T_a : numpy.ndarray or float
        Air temps in degrees C.
    T_s : numpy.ndarray or float
        Conductor surface temperatures (assumed to be sufficiently same as 
        conductor core temperature) in degrees C.
    D : numpy.ndarray or float
        Conductor diameter in m.
    emis : numpy.ndarray or float
        Conductor emissivity, unitless.
    absorp : numpy.ndarray or float
        Conductor absorptivity, unitless.
    R : numpy.ndarray or float
        Conductor resistance when surface is T_s in Ohms/m.
    W_v: numpy.ndarray or float
        Wind velocity, m/s.
    W_a : numpy.ndarray or float
        Wind direction (blowing into), clockwise degrees from due north
    elev : numpy.ndarray or float
        Elevation in m.
    air_qual : numpy.ndarray or float
        Value (0 to 1) indicating air quality; 0 = industrial, 1 = clear
    year : numpy.ndarray or int
        Year for rating calculations
    day_of_year : numpy.ndarray or int
        Day of the year for rating calculation (January 21 = 21; solstices on 
        172 and 355)
    start_hour : numpy.ndarray or int
        Time the rating hour starts (local standard time; do not adjust for
        daylight savings time) (0 to 23)
    latitude : numpy.ndarray or float
        Latitude of rating point, degrees
    longitude : numpy.ndarray or float
        Longitude of rating point, degrees
    time_zone : numpy.ndarray or int
        Time zone for rating location (relative to UTC; U.S. Pacific Standard 
        Time = â€“8 hours)
    line_azimuth : numpy.ndarray or float
        Angle between due north and the line, degrees clockwise from north
    verbose : int, optional (default = 0)
        If = 0, function returns only the line rating(s) (I).
        If = 1, function returns values for underlying calculation components 
          (q_c, q_r, q_s) in addition to line rating, each in W/m (I).
        If = 2, then also includes the following additional variable values in
          the output: I, q_c, q_c1, q_c2, q_cn, N_Re, k, visc, cond, dens, q_r,
                      q_s, year_fract, eqtime, decl, tst, ha, time_offset, 
                      zenith, sol_azimuth, Q_se, theta.
    
    Returns
    -------


    Numpy_array or int
        Array of calculated thermal transmission line rating in terms of
        current (Amps), rounded to nearest integer.
    When the value of verbose that is passed is 1 or 2, then additional 
        array/values (as discussed under "verbose" above) are returned.
    """
    
    #### Parameters ####
    
    # film temp, deg C (IEEE Eq. 6)
    T_film = (T_s + T_a)/2
    
    # density, kg/m**3 (IEEE Eq. 14a)
    dens = ((1.293 - 1.525E-4*elev + 6.379e-9*elev**2)
                /(1+0.00367*T_film)
            )
    
    # dynamic viscosity of air, kg/(m*s) (IEEE Eq. 13a)
    visc = 1.458e-6 * (T_film + 273)**1.5 / (T_film + 383.4)    
    
    # thermal conductivity of air, W/(m*degC) (IEEE Eq. 15a)
    cond = 2.424e-2 + 7.477e-5 * T_film - 4.407e-9 * T_film**2
    
    
    ####  Calculations ####
    
    # Reynolds number, dimensionless (IEEE Eq. 2c)
    N_Re = D * dens * W_v / visc
    
    # Angle difference between wind direction and line axis, radians (used
    # in the IEEE equations, but not defined)
    ang_diff = (2*np.pi/360) * (W_a - line_azimuth)
    
    # Wind direction factor, dimensionless (IEEE Eq. 4a)
    K = (1.194 - 
         np.cos(ang_diff) + 
         0.194*np.cos(2*ang_diff) + 
         0.368*np.sin(2*ang_diff)
         )
    
    # Low Reynolds number forced convection equation, W/m (IEEE Eq. 3a)
    q_c1 = K * (1.01 + 1.35 * N_Re**0.52 ) * cond * (T_s - T_a)
    
    # High Reynolds number forced convection equation, W/m (IEEE Eq. 3b)
    q_c2 = K * 0.754 * N_Re**0.6 * cond * (T_s - T_a)
    # Natural convection, W/m (IEEE Eq. 5a)
    q_cn = 3.645 * dens**0.5 * D**0.75 * (T_s - T_a)**1.25
    
    # Overal convection (highest of q_c1, q_c2, and q_cn)
    
    # First, assume highest of q_c1 and q_c2
    q_c = np.maximum(q_c1, q_c2)


    
    # Next, replace with q_cn if q_cn is higher than q_c1 or q_c2
    q_c = np.maximum(q_c, q_cn)
    
    # Radiated heat loss, W/m (IEEE Eq. 7a)
    q_r = (17.8 * D * emis * 
                (
                    ((T_s - -273)/100)**4 - 
                    ((T_a - -273)/100)**4
                )
            )
    
    # Solar heating
    
    # Check for leap year
    if np.isscalar(year): # if year passed as a scalar
        if year % 4 == 0:
            days = 366
        else: 
            days = 365
    else: # if year passed as a 1D array
        days = np.array([366 if yr % 4 == 0 else 365 for yr in year])
    
    # Fractional year, radians (NOAA at p. 1)
    year_fract = ( (2*np.pi/days) * 
                      (day_of_year - 1 + (start_hour - 12)/24)
                    )
    
    # Equation of time, minutes (NOAA at p. 1)
    eqtime = (229.18*
              (0.000075 + 
               0.001868*np.cos(year_fract) - 
               0.032077*np.sin(year_fract) - 
               0.014615*np.cos(2*year_fract) -
               0.040849*np.sin(2*year_fract)
              )
              )
    
    # Solar declination angle, radians (NOAA at p. 1)
    decl = (0.006918 -
            0.399912*np.cos(year_fract) +
            0.070257*np.sin(year_fract) - 
            0.006758*np.cos(2*year_fract) + 
            0.000907*np.sin(2*year_fract) - 
            0.002697*np.cos(3*year_fract) + 
            0.001480*np.sin(3*year_fract)
            )
    
    # Time offset, minutes (NOAA at p. 1)
    time_offset = eqtime + 4*longitude - 60*time_zone
    
    # True solar time, minutes (NOAA at p. 1)
    tst = start_hour*60 + time_offset
    


    # Solar hour angle (time, in degrees, between local point and solar noon), 
    # degrees (NOAA at p. 1)
    ha = (tst/4) - 180
    
    # Solar zenith angle (between sun's rays and vertical), radians (NOAA
    # at p. 1)
    zenith = np.arccos(
                        np.sin(2*np.pi*latitude/360) * np.sin(decl) +
                        np.cos(2*np.pi*latitude/360) * np.cos(decl) * 
                                                    np.cos(2*np.pi*ha/360)
                      )
    
    # Solar altitude angle (= 90 - zenith), radians (calculating since this is 
    # the variable (not zenith) used in the IEEE 738 calculation of solar 
    # heating)
    sol_altitude = (np.pi/2)-zenith
    
    # Solar azimuth (angle between due north and the shadow cast by
    # a vertical rod), radians clockwise from north (NOAA at p. 1))
    sol_azimuth = -np.arccos(
                       -(np.sin(2*np.pi*latitude/360) * np.cos(zenith) -
                                         np.sin(decl)) /
                        (np.cos(2*np.pi*latitude/360) * np.sin(zenith))
                        ) + np.pi
    
    # Solar heating
    
    # Solar heat flux for clear air at sea level, W/m**2 (IEEE Eq. 18)
    Q_s_clear = (
        (-42.2391     ) +
        ( 63.8044     ) * (sol_altitude*360/(2*np.pi)) +
        (- 1.9220     ) * (sol_altitude*360/(2*np.pi))**2 +
        (  3.46921e-2 ) * (sol_altitude*360/(2*np.pi))**3 +
        (- 3.61118e-4 ) * (sol_altitude*360/(2*np.pi))**4 +
        (  1.94318e-6 ) * (sol_altitude*360/(2*np.pi))**5 +
        (- 4.07608e-9 ) * (sol_altitude*360/(2*np.pi))**6
        )
    
    # Solar heat flux for industrial air at sea level, W/m**2 (IEEE Eq. 18)
    Q_s_industrial = (
        ( 53.1821    ) +
        ( 14.2110    ) * (sol_altitude*360/(2*np.pi)) +
        (  6.6138e-1 ) * (sol_altitude*360/(2*np.pi))**2 +
        (- 3.1658e-2 ) * (sol_altitude*360/(2*np.pi))**3 +
        (  5.4654e-4 ) * (sol_altitude*360/(2*np.pi))**4 +
        (- 4.3446e-6 ) * (sol_altitude*360/(2*np.pi))**5 +
        (  1.3236e-8 ) * (sol_altitude*360/(2*np.pi))**6
        )
    
    # Solar heat flux at sea level given local air quality, W/m**2
    # (step included to allow intermediate air qualities between
    # industrial and clear)
    Q_s = (Q_s_clear *      (air_qual    ) + 
           Q_s_industrial * (1 - air_qual)


           )
    
    # Candidate solar heat flux at given elevation, W/m**2 (IEEE Eq. 20)
    Q_se = (
        Q_s *
            (( 1.0      ) +
             ( 1.148e-4 ) * elev + 
             (-1.108e-8 ) * elev**2)
        )
    
    # If nighttime, then set Q_se to zero.  Per NOAA document, night is when
    # solar zenith >= 90.883 degrees (the approximate correction for 
    # atmospheric refraction at sunrise and sunset, and the size of the solar 
    # disk). (NOAA at p. 2)
    if np.isscalar(zenith):
        if zenith >= 90.883 * (2*np.pi/360):
            Q_se = 0
    else:
        Q_se = np.array( 
            [0 if zen >= 90.883 * (2*np.pi/360) 
             else qse 
             for (zen, qse) in np.array(list(zip(zenith,Q_se)))]
            )
    
    # Effective angle of incidence of sun's rays, radians (IEEE Eq. 9)
    theta = np.arccos(
                np.cos(sol_altitude) * 
                np.cos(sol_azimuth - (line_azimuth * 2 * np.pi / 360))
                )
        
    # Rate of solar heat gain in line, W/m (IEEE Eq. 8)
    q_s = absorp * Q_se * np.sin(theta) * D
    
    # Line rating, Amps (IEEE Eq. 1b)
    I = np.sqrt(
                (q_c + q_r - q_s)/R
                )
    
    # Return rating(s) and/or other calculated values
    # If argument "verbose" was passed to function as 1 or 2, then return
    # key underlying calculation values in a namedtuple; otherwise, just
    # return the rating(s).
    if verbose == 1:
        rating = namedtuple('output',['I', 'q_c', 'q_r', 'q_s'])
        return rating(np.rint(I),
                      np.around(q_c,3),
                      np.around(q_r,3),
                      np.around(q_s,3),
                      )
    elif verbose == 2:
        rating = namedtuple('output',['I',
                                      'q_c',
                                      'q_c1',
                                      'q_c2',


                                      'q_cn',
                                      'N_Re',
                                      'k',
                                      'visc',
                                      'cond',
                                      'dens',
                                      'q_r', 
                                      'q_s',
                                      'year_fract',
                                      'eqtime',
                                      'decl', 
                                      'tst',
                                      'ha',
                                      'time_offset',
                                      'zenith', 
                                      'sol_azimuth', 
                                      'Q_se', 
                                      'theta'])
        return rating(np.rint(I), 
                      np.around(q_c,3), 
                      np.around(q_c1,3),
                      np.around(q_c2,3),
                      np.around(q_cn,3),
                      np.around(N_Re,3),
                      np.around(K,3),
                      np.around(visc,8),
                      np.around(cond,4), 
                      np.around(dens,4),
                      np.around(q_r,3),
                      np.around(q_s,3), 
                      np.around(year_fract,3),
                      np.around(eqtime,4), 
                      np.around(decl,3), 
                      np.around(tst,3),
                      np.around(ha,2),
                      np.around(time_offset,2),
                      np.around(zenith,3), 
                      np.around(sol_azimuth,3), 
                      np.around(Q_se,3), 
                      np.around(theta,3),
                      )
    else:
        return np.rint(I)
    
def repackage_args(
    df_BranchAAR: pd.DataFrame,
    df_RatingPoints:pd.DataFrame) -> np.ndarray:
    """
    Function for reshaping fixed parameters for line rating calculations
    
    Allows an array of data to be fed into the ThermalRating functions one
    row at a time, for very fast computation.
    """


    ### Other Parameters ###
    n_points_types =  10539 # 10,539 total combinations of candidate rating point* 3 rating types
    ## Create arrays from dataframes (to speed math operations)
    columns = ['line','Dia m','Emis','Absorp','Line azimuth deg',
               'T_s_Norm C','T_s_LTE C','T_s_STE C',
                'R at T_Norm Ohm/m','R at T_LTE Ohm/m','R at T_STE Ohm/m']

    ar_BranchAAR = np.array(df_BranchAAR[columns])
    
    ar_RatingPoints = np.array(df_RatingPoints)
    
    ### create calc_arguments array to store argument data ###
    rows = 2529360 # 2,529,360 total combinations of line/point, type, and fhour
    shape = (rows,22) # line, point, type, fhour, rating
    calc_arguments = np.empty(shape=shape)
    calc_arguments[:] = np.nan
    
    ### cols 0 and 1 (line and point) ###
    # Simple repetition from top to bottom of array, in first two columns
    calc_arguments[:,0:2] = np.tile(ar_RatingPoints[:,[0,1]], (3*240,1))
    
    ### col 2 (rating types) ###
    calc_arguments[0             :int(rows/3),   2] = 0
    calc_arguments[int(rows/3)   :int(2*rows/3), 2] = 1
    calc_arguments[int(2*rows/3) :rows,          2] = 2
    
    ### col 3 (fhour) ###
    x = np.repeat(range(1,241),3513) # get array with 240 hours, with each hour repeated 3513*3 times
    x = x.reshape(3513*240,1) # reshape to a vertical array
    x = np.tile(x,(3,1))
    calc_arguments[:,3:4] = x
    
    # ### col 4 (T_a) ###
    ### leave this blank for now.  These column(s) are updated later in the demo
    
    ### cols 5 and 9 (T_s and R) ###
    # Populate T_s and R values at same time, since they both depend on rating type
    z = np.empty(shape=(3513*3,3)) # initiate array z to store calculated T_s and R values
    z[:] = np.nan
    j = 0 # initiate var to store row index for z
    for rt_idx, rating_type in [(0,'Norm'),(1,'LTE'),(2,'STE')]: # Loop through the rating types
        T_s_ix = 5 + rt_idx # looks up the appropriate allowable conductor temp for the rating type
        R_ix = 8 + rt_idx # looks up the appropriate resistance (at the allowable conducter temp) for the rating type
        for i in range(ar_RatingPoints.shape[0]): #loop through candiate rating points
            z[j,0] = int(rt_idx) # store index of z in column 0
            mask = ar_BranchAAR[:,0] == ar_RatingPoints[i,0] # create mask to filter rows in ar_BranchAAR to those matching the line number at index i
            z[j,1:] = ar_BranchAAR[mask,[T_s_ix, R_ix]] # store T_s and T_r values (fron ar_BranchAAR data) in columns 1 and 2 ofr z (at row j)
            j = j + 1 # increment j
    for rt_idx in [0,1,2]: # loop through rating types again (this time to save valus in calc_arguments)


        # assign 1/3 of the 2529360 rows (in cols 5 and 9, for T_s and R) at a time
        calc_arguments[int(rows*rt_idx/3):int(rows*(rt_idx+1)/3),[5,9]] = \
                        np.tile(z[int(n_points_types*rt_idx/3):int(n_points_types*(rt_idx+1)/3),1:],(240,1))
    
    ## the next two blocks of code are fairly inefficient, and should be improved.
    
    ### cols 6, 7, 8, 20 (D, emis, absorp, line azimuth) ###
    range_ = np.arange(calc_arguments.shape[0], dtype=np.int64)
    for i in range(ar_BranchAAR.shape[0]):
        mask = calc_arguments[:, 0] == ar_BranchAAR[i, 0]
        if np.count_nonzero(mask) != 0:
            true_counts = np.count_nonzero(mask)
            broadcast_ar_BranchAAR = np.broadcast_to(ar_BranchAAR[i, [1, 2, 3, 4]], shape=(true_counts, 4))
            broadcast_calc_arguments = np.broadcast_to([6, 7, 8, 20], shape=(true_counts, 4))
            calc_arguments[range_[mask][:, None], broadcast_calc_arguments] = broadcast_ar_BranchAAR
    
    ### cols 10, 11, 12, 13 (W_v, W_a, elev, air_qual) and 17, 18, 19 (latitude, longitude, time_zone) ###
    range_ = np.arange(calc_arguments.shape[0], dtype=np.int64)
    for i in range(ar_RatingPoints.shape[0]):
        print(f'\rProcessing data for point {i+1} of 3513 ', end='')
        mask1 = calc_arguments[:,0] == ar_RatingPoints[i,0]
        mask2 = calc_arguments[:,1] == ar_RatingPoints[i,1]
        mask = np.logical_and(mask1,mask2)
        if np.count_nonzero(mask) != 0:
            true_counts = np.count_nonzero(mask)
            broadcast_ar_RatingPoints = np.broadcast_to(ar_RatingPoints[i, [7,8,4,6,2,3,5]], shape=(true_counts, 7))
            broadcast_calc_arguments = np.broadcast_to([10,11,12,13,17,18,19], shape=(true_counts, 7))
            calc_arguments[range_[mask][:, None], broadcast_calc_arguments] = broadcast_ar_RatingPoints
    
    # # ### cols 14, 15, 16 (year, day_of_year, start_hour) ###
    ### leave this blank for now.  These column(s) are upated later in the demo
    
    print('complete')
    
    return calc_arguments
def update_args(calc_arguments, adjustedTemp, initDateTime):
    """
    This function adds temperature and date/time data to calc_arguments array
    to prepare the array for sending to ThermalRatings() to compute line
    ratings(s)
    """
    ### col 4 (T_a) ###
    x = np.reshape(adjustedTemp,(3513*240,1),order='F') # first unravel and stack the columns from adjustedTemp
    x = np.tile(x,(3,1)) # repeat that vertically 3 times (once for each rating type)
    calc_arguments[:,4:5] = x
    # ### cols 14, 15, 16 (year, day_of_year, start_hour) ###
    # get unique values of time_zone from ar_Rating_Points
    timeZones = np.unique(calc_arguments[:,19])
    for timeZone in timeZones: #loop over those time zone values
        # First generate the correct values for each value of fhour up to user-specified numHours
        numHours = 240


        # Initialize an array to store fhour, local_year, local_day_of_year, local_start_hour
        fhour = np.empty(shape = (numHours,4))
        fhour[:] = np.nan
        for fhour in range(1,numHours+1):
            print('\rProcessing weather and time references for forecast hour f{}'.format(str(fhour).zfill(3)),end='')
            range_ = np.arange(calc_arguments.shape[0], dtype=np.int64)
            forecastTime = initDateTime + timedelta(hours=fhour+int(timeZone))
            local_year= int(forecastTime.strftime('%Y'))
            local_day_of_year = int(forecastTime.strftime('%-j'))
            local_start_hour = int(forecastTime.strftime('%H'))
            mask1 = calc_arguments[:,3] == fhour
            mask2 = calc_arguments[:,19] == timeZone
            mask = np.logical_and(mask1,mask2)
            if np.count_nonzero(mask) != 0:
                true_counts = np.count_nonzero(mask)
                broadcast_fhours = np.broadcast_to([local_year, local_day_of_year, local_start_hour], shape=(true_counts, 3))
                broadcast_calc_arguments = np.broadcast_to([14,15,16], shape=(true_counts, 3))
                calc_arguments[range_[mask][:, None], broadcast_calc_arguments] = broadcast_fhours
    
    print('\ncomplete')
    return
