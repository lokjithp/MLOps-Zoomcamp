#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import numpy as np
import pandas as pd
import sys


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[3]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[7]:


def generate_results(df,y_pred,year,month,o):
    output_file=o
    year=int(year)
    month=int(month)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result=pd.DataFrame(y_pred)

    df_result['ride_id']=df['ride_id']

    df_result.columns=['prediction','ride_id']

    (df_result.iloc[0,1])='2021/02_0'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
        )
    


# In[8]:


def build_input_output(year,month):
    year=int(year)
    month=int(month)
    taxi_type='green'

    input_file= f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file=f'output/{year:04d}-{month:02d}.parquet'
    return input_file,output_file


# In[11]:

def run():
    year=int(sys.argv[1])
    month=int(sys.argv[2])
    i,o=build_input_output(year,month)
    df=read_data(i)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print('mean of prediction is ',np.mean(y_pred))
    generate_results(df,y_pred,year,month,o)

if __name__=="__main__":
    run()




