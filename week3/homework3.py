from asyncio.log import logger
from sysconfig import get_paths
import pandas as pd
from pendulum import today

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task,get_run_logger
from prefect.task_runners import SequentialTaskRunner

import datetime
import pickle
import mlflow

from sklearn.model_selection import train_test_split


import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df
@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    logger=get_run_logger()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
@task
def train_model(df, categorical):
    logger=get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")
    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    logger.info("The MSE of training is: {mse}")
    return lr, dv
@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values
    logge=get_run_logger()

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return
@task
def get_paths(date=None):
    today=datetime.datetime.now()
    if(date == None):
        date= today.strftime("%Y-%m-%d")
        print(date)
        cur_month=today.month
        print(cur_month)
    else:
        given_date=datetime.datetime.strptime( date,"%Y-%m-%d")
        cur_month= given_date.month
    if(cur_month-1<=0):
        prev_month=12
    else:
        prev_month=cur_month-1
    if(prev_month-1<=0):
        prev2_month=12
    else:
        prev2_month=prev_month-1

    train_path='./data/fhv_tripdata_2021-0'+str(prev2_month)+'.parquet'
    val_path='./data/fhv_tripdata_2021-0'+str(prev_month)+'.parquet'
    return train_path,val_path
    
#train_path: str = './data/fhv_tripdata_2021-01.parquet', 
 #          val_path: str = './data/fhv_tripdata_2021-02.parquet'

@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    train_path,val_path=get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)
    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open("models/model-"+date+".b", "wb") as f_out:
        pickle.dump(lr, f_out)
    with open("models/dv-"+date+".b","wb") as dv_out:
        pickle.dump(dv,dv_out)
    

#main(date="2021-08-15")



from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron=" 0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["homework 3"]
)

#schedule=CronSchedule(cron=" 0 9 15 * *")
