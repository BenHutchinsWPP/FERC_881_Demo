REM You may have issues running this in Windowd due to PyGrib.
REM Internet searches indicate that running in Linux may be preferred,
REM but PyGrib might work in Windows under certain circumstances now 
REM using an Anaconda setup. An attempt to setup the environment properly
REM is below but may require more tweaking to make this run in Windows.

conda install pip
conda install -c conda-forge pygrib=2.0.1
pip install -r requirements.txt
