import pandas as pd
import numpy as np
import geopandas
import contextily as ctx
import matplotlib.pyplot as plt
from natsort import index_natsorted, natsorted, natsort_keygen

import glob,os

for filepath in glob.iglob('./original/*.txt'):
    fdata = pd.read_csv(filepath, sep='\t',names=['cellid','time','countrycode','smsin','smsout','callin','callout','internet' ])
    fdata['time']=pd.to_datetime(fdata['time']+3600000, unit='ms') #+3600000 to change timezone
    filt=fdata.sort_values(
        by=["cellid","time"],
        key=natsort_keygen()
    )
    filt.to_csv(os.path.splitext(filepath)[0]+'.csv')
    print("Processed:"+os.path.splitext(filepath)[0])

#fdata = pd.read_csv('sms-call-internet-mi-2013-11-01.txt', sep='\t',names=['cellid','time','countrycode','smsin','smsout','callin','callout','internet' ])