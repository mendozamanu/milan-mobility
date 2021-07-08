import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dfs = pd.DataFrame({})

for i in range(1,10):
    df = pd.read_csv('./milan/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['time'])
    dfs = dfs.append(df)
for i in range(10,31):
    df = pd.read_csv('./milan/sms-call-internet-mi-2013-11-{}.csv'.format(i), parse_dates=['time'])
    dfs = dfs.append(df)
for i in range(1,10):
    df = pd.read_csv('./milan/sms-call-internet-mi-2013-12-0{}.csv'.format(i), parse_dates=['time'])
    dfs = dfs.append(df)
for i in range(10,32):
    df = pd.read_csv('./milan/sms-call-internet-mi-2013-12-{}.csv'.format(i), parse_dates=['time'])
    dfs = dfs.append(df)
df = pd.read_csv('./milan/sms-call-internet-mi-2014-01-01.csv', parse_dates=['time'])
dfs = dfs.append(df)
dfs = dfs.fillna(0)
del df
#Group by hours and aggregate values according to each cell/hour
dfgr = dfs[['cellid', 'time', 'smsin','smsout', 'callin','callout', 'internet']].groupby(['time', 'cellid'], as_index=False).sum()
dfgr['hour'] = dfgr.time.dt.hour+24*(dfgr.time.dt.day-1)+((30*24)*(dfgr.time.dt.month-11))+(361*24*(dfgr.time.dt.year-2013))
dfgg = dfgr[['hour', 'cellid', 'time', 'smsin','smsout', 'callin','callout', 'internet']].groupby(['hour', 'cellid'], as_index=False).sum()
#dfgg = dfgg.set_index(['hour']).sort_index()
del dfs
#20x20 grid
l=[] #choosen cells on 20x20 grid
for i in range(4263,6173,100):
    for j in range(i,i+20):
        l.append(j)
#Start creating the 20x20 dataframe
df20 = pd.DataFrame({})
for el in dfgg.iterrows():
    if(l.count(int(el[1].cellid))==1):
        tmp = pd.DataFrame([[int(el[1].hour), int(el[1].cellid), el[1].smsin, el[1].smsout, el[1].callin, el[1].callout, el[1].internet]], columns=['hour', 'cellid', 'smsin','smsout', 'callin','callout', 'internet'])
        df20=df20.append(tmp)
df20 = df20.set_index(['hour']).sort_index()
cells = df20.cellid
scaler=MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(df20), columns=df20.columns, index=df20.index)
dfgg = scaled
dfgg['cellid']=cells
dfgg.to_csv('./results/mi_temp_mon.csv')
#temp save of the current data

#complete the dataframe
dfsi = [] #data from df without indexing
dfsi2 = pd.DataFrame({}) #new df to save
dfso = []
dfso2 = pd.DataFrame({})
dfci = []
dfci2 = pd.DataFrame({})
dfco = []
dfco2 = pd.DataFrame({})
dfin = []
dfin2 = pd.DataFrame({})

dfsi.append(dfgg.loc[0].cellid)
dfso.append(dfgg.loc[0].cellid)
dfci.append(dfgg.loc[0].cellid)
dfco.append(dfgg.loc[0].cellid)
dfin.append(dfgg.loc[0].cellid)

dfsi2["cellid"]=dfsi[0].values
dfso2["cellid"]=dfsi[0].values
dfci2["cellid"]=dfsi[0].values
dfco2["cellid"]=dfsi[0].values
dfin2["cellid"]=dfsi[0].values
print (dfgr.hour.max())
for i in range(0, dfgr.hour.max()+1): #dfgr.hour.max()+1
    try:
        #row[0] - hour, row[1]: df cols
        dfsi.append(dfgg.loc[i].smsin)
        dfsi2["smsin"+str(i)]=dfsi[i+1].values
        
        dfso.append(dfgg.loc[i].smsout)
        dfso2["smsout"+str(i)]=dfso[i+1].values
        
        dfci.append(dfgg.loc[i].callin)
        dfci2["callin"+str(i)]=dfci[i+1].values
        
        dfco.append(dfgg.loc[i].callout)
        dfco2["callout"+str(i)]=dfco[i+1].values
        
        dfin.append(dfgg.loc[i].internet)
        dfin2["internet"+str(i)]=dfin[i+1].values
    except: #si para alguna hora hay alguna celda sin datos, continuar a la sig hora
        continue

dfsi2.to_csv("./results/mi/classif20x20month_smsin.csv")
dfso2.to_csv("./results/mi/classif20x20month_smsout.csv")
dfci2.to_csv("./results/mi/classif20x20month_callin.csv")
dfco2.to_csv("./results/mi/classif20x20month_callout.csv")
dfin2.to_csv("./results/mi/classif20x20month_internet.csv")
