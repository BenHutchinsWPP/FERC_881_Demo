#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains functions for downloading NOAA NOMADS weather data.
"""
import requests, time, pygrib, os, shutil
from sys import getsizeof
from dask import delayed, compute
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
def Temp2m(
    previousHours = 1,
    forecastHourSet = range(1,268), 
    leftlon = -120,
    rightlon = -110,
    toplat = 38,
    bottomlat = 31, 
    saveDirectory = 'NBM/'):
    """
    Function for downloading National Blend of Models (NBM) 2m surface
    temperature data.
    
    Note:  At times the NOAA server can fail to notify the client if a 
    particular files is unavailable.  In that instance, this function can loop 
    and eventually time out.  Note that this does not affect the successful 
    download of files that are available.
    Parameters
    ----------
    previousHour : int, optional
        Specification of the initiation hour to download, relative to current 
        hour.  For example, if the current UTC hour is 5 and previousHour is
        provided as 2, then the function will download NBM files with the
        initiation hour 5 - 2 = 3 UTC. The default is 1.
    forecastHourSet : list, optional
        List of hours (or a range object) specifying which forecast hours to
        attempt to download. The default is range(1,268).
    leftlon : int, optional
        Longitude specifying the left boundry of area for which to download 
        weather data. The default is -120.
    rightlon : int, optional
        Longitude specifying the right boundry of area for which to download 
        weather data.  The default is -110.
    toplat : int, optional
        Latitude specifying the top boundry of area for which to download 
        weather data.  The default is 38.
    bottomlat : int, optional
        Latitude specifying the bottom boundry of area for which to download 
        weather data.  The default is 31.
    saveDirectory : str, optional


        Directory where the grib2 files will be saved. The default is 'NBM/'.
    Returns
    -------
    now : datetime
        Datetime object reflecting the initiation time for the forecast
        downloaded.
    initHour : int
        Initiation hour for the forecast downloaded.
    successHours : list
        List of ints representing the forecast hours for which files were
        successfully downloaded.
    """
    
    # Clear files from saveDirectory
    for files in os.listdir(saveDirectory):
        path = os.path.join(saveDirectory, files)
        try:
            shutil.rmtree(path)
        except OSError: #if try command doesn't work on local OS
            os.remove(path)
        
    # Create list to hold status of downloads for each forecast hour
    status = [0]*len(forecastHourSet)
    # Store initiation date/time
    now = datetime.utcnow() - timedelta(hours=previousHours) #adjust for previousHours
    initHour = now.strftime('%H')
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')
    
    # Record start time to allow timeout, and for various print statements
    startTime = time.time()
    
    print('--------------------------------------------------------------------------------')
    print('Downloading NOAA National Blend of Models (NBM) 2m air temperature data from the')
    print('NOAA NOMADS server at https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl.')
    print('-- Forecast initialization date and hour:  {}/{}/{} {}h UTC'.format(day,month,year,initHour))
    print('-- Download process started at {} local time'.format(time.ctime(startTime)))
    print('--------------------------------------------------------------------------------')
    print('\n')
    
    # Initiate var to store attempt #
    attempt = 0
    
    # Do while the status list contains any entries that are 0 or 302
    # (0 = download has not been attempted yet; 302 = download attempt was
    #  unsuccessful (and no information was retrieved from the server))
    while (0 in status) or (302 in status):
        
        # If the current time is beyond the timeout time, then abort with a message
        if time.time() > startTime + 60*4: print('--------- Script aborted; exceeded max time. ---------'); return


        
        attempt = attempt + 1 # increment attempt, and print status
        print('Attempt #{} starting {}'.format(attempt, time.ctime(time.time())))
        
        # Call setup_and_loop to attempt to download files and recored their download statuses.
        # Cast output as a list (from the returned tuple), because tuples do not support the item 
        # assignment that may happen in future loops/calls to setup_and_loop().
        status = list(setup_and_loop(forecastHourSet, status, initHour, leftlon, rightlon, toplat, bottomlat, year, month, day, saveDirectory))
        
        # Print status
        print('Attempt #{} results shown below:'.format(attempt))
        printStatus(status)
        
        # To avoid very rapid requests for the same files (which some servers can object to), sleep
        # a few seconds between loops if there are less than 10 files remaining to download.
        unsuccessful_hours = [index + 1 for index,value in enumerate(status) if value == 302]
        if len(unsuccessful_hours) < 10: time.sleep(5)
            
    # Print that download was successful, and indicate completion time
    print('-------------------------------------------------------------------------------------------')
    print('Download successfully completed at {}. All download statuses are 200'.format(time.ctime(time.time())))
    print('(successful) or 404 (no file on server).')
    print('-------------------------------------------------------------------------------------------')
    
    # Identify forecast hours with successfully downloaded data files (to be returned)
    successHours = [idx + 1 for idx, x in enumerate(status) if x == 200]
    
    # Return now, initHour and successHours (to be used in next steps of AAR calculation process)
    return now, initHour, successHours

def printStatus(status):
    """Prints a table of download statuses"""
    
    # Convert http status codes to symbols (for display in table)
    code_status = [' + ' if x == 200 else ' â‹… ' if x == 404 else '***' for x in status]
    
    # Construct/print table of statuses
    print('\nforecast   ( + = success,  *** = still trying,')
    print('hour             â‹… = no file on server)')
    print('decade:   --------------  status ---------------\n')
    for i in range(0, len(status), 10):
        time.sleep(0.05)
        print(str(i).zfill(3) + ':      ', end='')
        print(*code_status[i:i+10])
    print('\n\n')

def setup_and_loop(forecastHourSet, status, initHour, leftlon, rightlon, toplat, bottomlat, year, month, day, saveDirectory):
    """Loops over forecast hours and executes execRequestSave() to download/save files"""
        
    limit_count = 0


    
    # loop over forecast hours
    for index, hour in enumerate(forecastHourSet):
        
        # If the status associated with this hour is 200 (complete) or 404 
        # (not available), then skip to next hour
        if status[index] in {200, 404} or limit_count > 99:
            continue
        
        limit_count = limit_count + 1
        
        # Setup arguments for calling execRequestSave()        
        url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_blend.pl'
        filename = 'blend.t' + str(initHour).zfill(2) + 'z.core.f' + str(forecastHourSet[index]).zfill(3) + '.co.grib2'
        params = {'file': filename,
                  'lev_2_m_above_ground':'on',
                  'var_TMP':'on',
                  'subregion':'',
                  'leftlon': str(leftlon),
                  'rightlon': str(rightlon),
                  'toplat': str(toplat),
                  'bottomlat': str(bottomlat),
                  'dir':'/blend.' + str(year) + str(month).zfill(2) + str(day).zfill(2) +'/' + 
                      str(initHour).zfill(2) + '/core'
                  }
        
        #execRequestSave(url, params, filename)
        # Use dask delayed() to parallelize the execution (allows multi-core computers
        # to download much faster than serial execution).  This statement is similar to
        # calling execRequestSave(url, params, filename), but instead of executing immediately,
        # it creates a "lazy" command that is not executed until the compute() command is
        # executed below (outside of the loop).
        status[index] = delayed(execRequestSave)(url, params, filename, saveDirectory)
        
        
        
    # dask compute() statement, to start parallel operation of steps created above
    return compute(*status)

def execRequestSave(url,params,filename, saveDirectory):
    """Executes the requests.get command, and saves the results if successful"""
    
    # Execute requests.get.  If times out, then return 302 to indicate error.
    while True:
        try:
            r = requests.get(url,params=params,timeout=(20,20))
            break
        except:
            return 302
    
    # Extract size from r.content, to help determine if file was successfully downloaded.
    # Sometimes the server returns a file saying that the data is not available, and
    # usually it returns a code of 404 with that response.  But occassionally the


    # server returns a code of 200 for those responses.  Those responses have
    # smaller file sizes {~0.4kb} where as the grib2 files all have files sizes
    # above 150kb.
    filesize = getsizeof(r.content)
    
    # If the file was successfully downloaded, then save it
    if r.status_code == 200 and filesize > 150000:
        with open(saveDirectory + filename, 'wb') as f:
            f.write(r.content)
        return 200
    #Iif the status code is 200, but the file size isn't large enough to be a successful
    # download, then don't save and return 302 to indicate an error
    elif r.status_code == 200:
        return 302 
    #Iif anything else, then don't save and just return that status code
    else:
        return r.status_code

def process_nbm_grib_files(initHour, forecastHourSet, RatingPointsIJs, dataDirectory = 'NBM/'):
    """Function to process grib2 files"""
    
    # Initialize numpy arrays to store temperature and standard deviation data
    temperatureByIndex    = np.empty((len(RatingPointsIJs), len(forecastHourSet)))   ## 3513 x 97
    temperatureSTDbyIndex = np.empty((len(RatingPointsIJs), len(forecastHourSet)))   ## 3513 x 97
    temperatureByIndex[:] = np.nan
    temperatureSTDbyIndex[:] = np.nan    
    
    # Loop through forecastHourSet, and open the grib2 files for each hour, and
    # extract their vlaues to numpy arrays temperature and temperatureSTD
    for idx, forecastHour in enumerate(forecastHourSet):
        grib = '{}blend.t{}z.core.f{}.co.grib2'.format(dataDirectory,str(initHour).zfill(2), str(forecastHour).zfill(3))
        grbs = pygrib.open(grib)
        temperature = grbs[1].values # extract temperature forecast values (deg K) ## dimension of NBM grid
        temperatureSTD = grbs[2].values # extract forecast standard deviation values (deg C)  ## dimension of NBM grid
    
        # Convert from Kelvin to Celsius
        temperature = temperature - 273.15
        
        # Match temperature and standard deviation to each index location
        for indexNum in range(0, len(RatingPointsIJs)):  # num of rating points 3513
            temperatureByIndex[indexNum, idx] = temperature[RatingPointsIJs[indexNum, 0], RatingPointsIJs[indexNum, 1]]  ## 3513 x ~97
            temperatureSTDbyIndex[indexNum, idx] = temperatureSTD[RatingPointsIJs[indexNum, 0], RatingPointsIJs[indexNum, 1]]  ## 3513 x ~97
    
    #Create arrays with all hours and interpolate values between foreacast hours
    allHours = np.arange(1,np.amax(forecastHourSet),1) ## 240 x 1
    
    # Initialize numpy arrays to store temp and STD data by index and hour
    temperatureByIndexAndHour = np.empty((len(RatingPointsIJs), len(allHours)))  ## 3513 x ~240
    temperatureSTDbyIndexAndHour = np.empty((len(RatingPointsIJs), len(allHours)))  ## 3513 x ~240
    temperatureByIndexAndHour[:] = np.nan
    temperatureSTDbyIndexAndHour[:] = np.nan


    
    #Loop over allHours and add data to arrays
    for idx, forecastHour in enumerate(allHours): # ~240
        # If the temp/std data were already downloaded from NBM/NOAA, then just recored that data (no need to interpolate)    
        if forecastHour in forecastHourSet:
            temperatureByIndexAndHour[:, idx] = temperatureByIndex[:, forecastHourSet.index(forecastHour)]
            temperatureSTDbyIndexAndHour[:, idx] = temperatureSTDbyIndex[:, forecastHourSet.index(forecastHour)]
    
    # If the temp/std data are missing, then interpolate for missing hours.  Note that this method was used to create
    # data to use in the demo, but is not methodologically valid.
    for idx in range(temperatureByIndexAndHour.shape[0]): #3513
        # replace each temperature row with a row sent to Pandas for interpolation
        temperatureByIndexAndHour[idx] = np.array(pd.Series(temperatureByIndexAndHour[idx]).interpolate())
        temperatureSTDbyIndexAndHour[idx] = np.array(pd.Series(temperatureSTDbyIndexAndHour[idx]).interpolate())
    
    #Round to one decimal place
    temperatureByIndexAndHour[:] = np.round(temperatureByIndexAndHour, 1)
    temperatureSTDbyIndexAndHour[:] = np.round(temperatureSTDbyIndexAndHour, 1)
    
    # return all values (even for fhours > 239); let the user decide what fhours to keep or drop depending on when they pulled the forecast data
    return temperatureByIndexAndHour, temperatureSTDbyIndexAndHour