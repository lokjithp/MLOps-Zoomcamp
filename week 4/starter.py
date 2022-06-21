from calendar import month
import pickle
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

with open('homework-4/model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')



dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)
print(y_pred)


print(np.mean(y_pred))
year=2021
month=2

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result=pd.DataFrame(y_pred)

df_result['ride_id']=df['ride_id']

df_result.columns=['prediction','ride_id']

(df_result.iloc[0,1])='2021/02_0'
print(df_result)

# df_result.to_parquet(
#     'output_file',
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
