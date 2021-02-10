''' TO DO LIST
err_list 파일 자동화
col_list
파일 폴더 자동 생성
더러운 코드 정리
hourstep mode(all/낮추출)
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import fnmatch
import math
from sklearn.compose import ColumnTransformer
from scipy.stats import yeojohnson
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PowerTransformer
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, RepeatVector, LSTM, Input, TimeDistributed, Activation, Dropout
from sklearn.compose import ColumnTransformer
#np.set_printoptions(suppress=True)

#variables
powhr_start = 5
powhr_end   = 20
shift_days  = 3
hoursteps   = powhr_end-powhr_start+1 #(16)
timesteps   = shift_days*hoursteps #hours step
data_dim    = 7
out_dim     = 1
n_model     = 2

#data_dir   = '../Data'
data_dir  = '../data/yoons'
season_mod = 'all_1102_f7'
date_start = '10190901'
date_end   = '30191201'
err_date_list =['20190912',
                '20191122',
                '20191130',
                '20191217',
                '20200501',
                '20200502',
                '20191028',
                '20191107',
                '20191108',
                '20191109',
                '20191110',
                '20191111',
                '20191112',
                '20200214',
                '20200307',
                '20200308',
                '20200309',
                '20200310',
                '20200328',
                '20200329',
                '20200625',
                '20200809']


#############################################
# Transformation
#############################################
def transform(transformMode, dataset):
    print("**********************pvBackend>transform>", transformMode)
    print("dataset :\n", dataset)

    if("1"==transformMode):#log_transform
        print("log_transform >>")
        transformed_df = dataset.apply(lambda x: np.log(x+1))
        
    elif("2"==transformMode):#squreroot_transform
        print("squreroot_transform >>")
        transformed_df = dataset.apply(np.sqrt)
        
    elif("3"==transformMode):#yeo_johnson_transform
        print("yeo-johnson >>")
        colList = []
        for col in dataset.columns: 
            colList.append((col, PowerTransformer(method='yeo-johnson', standardize=True), [col]))

        print("colList : ", colList)
        column_trans = ColumnTransformer(colList)
        transformed_data = column_trans.fit_transform(dataset)
        transformed_df = pd.DataFrame(transformed_data, columns=dataset.columns)
        pd.concat([transformed_df], axis = 1)

    elif("4"==transformMode):#boxcox_transform ------------------ TO BE
        print("boxcox_transform >>")
        transformed_df = dataset.apply(lambda x: np.log(x+1))
        colList = []
        for col in dataset.columns: 
            colList.append((col, PowerTransformer(method='box-cox', standardize=True), [col]))

        print("colList : ", colList)
        column_trans = ColumnTransformer(colList)
        transformed_data = column_trans.fit_transform(dataset)
        transformed_df = pd.DataFrame(transformed_data, columns=dataset.columns)
        pd.concat([transformed_df], axis = 1)

    elif("5"==transformMode):#wavelet_transform ------------------ TO BE
        print("wavelet_transform >>")
        colList = []
        for col in dataset.columns: 
            colList.append((col, PowerTransformer(method=transf_type, standardize=True), [col]))

        print("colList : ", colList)
        column_trans = ColumnTransformer(colList)
        transformed_data = column_trans.fit_transform(dataset)
        transformed_df = pd.DataFrame(transformed_data, columns=dataset.columns)
        pd.concat([transformed_df], axis = 1)
                            
    print(transformed_df)
    return transformed_df

#############################################
# photovoltaic power data
#############################################
def get_pow():
    
    # pow 파일 load
    dir_path    = data_dir+"/pow_24/UR00000126_csv"
    file_list   = os.listdir(dir_path)
    print(len(file_list))
    hrPow  = []    

    # pow측정값 에러가 큰 일자 제거
    for filename in file_list:
        if (filename[:-4] not in err_date_list):
            if ((filename[:-4]>=date_start) & (filename<date_end)):
                filedata = pd.read_csv(dir_path+'/'+filename).values[:,0]
                hrPow.append(filedata)
                
    #낮시간 추출 (5~20시)
    pow_dataset = pd.DataFrame(hrPow)
    pow_dataset =pow_dataset.iloc[:,powhr_start:powhr_end+1]
    #pow_dataset.to_csv("C:/Users/VISLAB_PHY/Desktop/WORKSPACE/Origin/data/pow_hr.csv",mode='w',index=False)

    # 결측값 보간, reshape
    pow_dataset = pow_dataset.interpolate(method='linear')
    pow_dataset = pow_dataset.values.reshape(-1,1)
    pow_dataset = pd.DataFrame(pow_dataset)
    pow_dataset.columns = ['pow']
    pow_dataset.to_csv(data_dir+"/pow.csv",mode='w',index=False)
    
    # scale
    sc_pow = MinMaxScaler(feature_range = (0, 1))
    scaled_pow = sc_pow.fit_transform(pow_dataset.values)
    df_pow = pd.DataFrame(scaled_pow, columns=pow_dataset.columns, index=list(pow_dataset.index.values))
    
    return df_pow, sc_pow

#############################################
# 종관기상관측
#############################################
def get_weather():
    # pow 파일 load
    file_list   = os.listdir(data_dir)
    print(len(file_list))
    for filename in os.listdir(data_dir):
        if fnmatch.fnmatch(filename, 'OBS_ASOS_TIM_*.csv'):
            print(filename)

            # load csv data
            dataset = read_csv(data_dir+'/'+filename, encoding='CP949')
            dataset.drop(['지점','지점명'], axis=1, inplace=True)
            dataset.drop(['기온 QC플래그','강수량 QC플래그','풍속 QC플래그','풍향 QC플래그','습도 QC플래그'], axis=1, inplace=True)
            dataset.drop(['현지기압 QC플래그','해면기압 QC플래그','일조 QC플래그','지면온도 QC플래그'], axis=1, inplace=True)
            dataset.drop(['5cm 지중온도(°C)','10cm 지중온도(°C)','20cm 지중온도(°C)','30cm 지중온도(°C)'], axis=1, inplace=True)
            dataset.drop(['3시간신적설(cm)','일사(MJ/m2)','운형(운형약어)','지면상태(지면상태코드)','현상번호(국내식)'], axis=1, inplace=True)

            # set column name
            dataset.columns = ['ymdhms', 'temprt', 'rain', 'wnd_spd', 'wnd_dir', 'humdt','steampressr',
                            'dewpnt', 'pressr','seapressr','sunshine','snow','cloud','cloud2','mincloud','visiblt','grd_temprt']

            # prioirty sort (피어슨상관계수)
            dataset = dataset[['ymdhms','sunshine','humdt','wnd_spd','visiblt','cloud2',
                            'cloud','grd_temprt','wnd_dir','dewpnt','steampressr','temprt',
                            'mincloud','rain','pressr','seapressr','snow']]


            # set NA data (관측값 0이 누적되어 결측된 경우. 0으로 세팅)
            dataset['rain'].fillna(0, inplace=True)     #강수량
            dataset['sunshine'].fillna(0, inplace=True) #일조
            dataset['snow'].fillna(0, inplace=True)     #적설량

            #일시 패턴 변환(2019-08-20 5:00 -> 2019082005)
            dataset['ymdhms'] = dataset['ymdhms'].str[0:4]+dataset['ymdhms'].str[5:7]+dataset['ymdhms'].str[8:10]+dataset['ymdhms'].str[11:13]
            # pow측정값 중 결측값 많은 일자 제거
            dataset = dataset[(dataset['ymdhms'].str[0:8]>=date_start) & (dataset['ymdhms'].str[0:8]<date_end)]
            for err_date in err_date_list:
                idx_err = dataset[dataset['ymdhms'].str.startswith(err_date)].index
                dataset = dataset.drop(idx_err)

            #낮시간 추출 (5~20시)
            dataset = dataset[(dataset['ymdhms'].str[-2:]>=str(powhr_start).rjust(2, '0')) &(dataset['ymdhms'].str[-2:]<=str(powhr_end))]
            dataset = dataset.interpolate(method='linear')# 결측값 보간
            
            # save file (test용)
            dataset.to_csv(data_dir+"/weather.csv",mode='w',index=False)

            # normalization
            dataset.drop(['ymdhms'], axis=1, inplace=True)
            dataset = dataset.astype('float32')
            dataset = dataset.interpolate(method='linear')
            
            #YEO-JOHNSON transform
            #yeo_df = yeo_johnson_transform(dataset)
            yeo_df = dataset
            
            
            #insert feature (test)
            yeo_df.insert(2, 'temp_press', yeo_df['temprt']-yeo_df['steampressr'], True)
            yeo_df.insert(2, 'sunshine_humdt', abs(yeo_df['sunshine'])-(yeo_df['humdt']*(2.1)), True)#0.35
            
            sc = MinMaxScaler(feature_range = (0, 1))#scale
            scaled_weather = sc.fit_transform(yeo_df.values)
            weather = pd.DataFrame(scaled_weather, columns=yeo_df.columns, index=list(yeo_df.index.values))
            print("before : ", weather.shape)
            weather = weather.iloc[:, 0:data_dim] #feature size 조절
            print("after : ", weather.shape)
            
    return weather

#############################################
# Deep Learning Model
#############################################
def trainDlModel():
    # get pow, weather
    df_pow, sc_pow   = get_pow()
    df               = get_weather()

    # df = pow + weather + powY
    df.insert(0, 'pow', df_pow.values, True)
    df = df.iloc[0:-timesteps, :]
    df.insert(df.shape[1], 'pow_Y', df_pow.iloc[timesteps:, :].values, True)
    df.to_csv(data_dir+"/total.csv",mode='w',index=False, encoding='CP949')

    # time step만큼 window 움직여 dataset 생성
    totalsize = df.shape[0]
    dataX, dataY = [], []

    for i in range(0, totalsize-timesteps-24+1, hoursteps):
        dataX.append(df.iloc[i:(i + timesteps),0:-1])
        dataY.append(df.iloc[i:(i + hoursteps),[0]])

    print("len(dataX) : ", len(dataX), dataX[0].shape)
    print("len(dataY) : ", len(dataY), dataY[0].shape)

    #  Split train/test  -> np.save
    #(****hy : train 비율 param으로 다시 잡기)
    train_size = int(len(dataX) * 0.7)
    val_size   = int(len(dataX) * 0.2)
    test_size  = len(dataX) - train_size - val_size
    val_idx = train_size+val_size

    trainX, valX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:val_idx]), np.array(dataX[val_idx:val_idx+test_size])
    trainY, valY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:val_idx]), np.array(dataY[val_idx:val_idx+test_size])

    print('train X : ', trainX.shape, '\tY : ', trainY.shape)
    print('val   X : ', valX.shape,   '\tY : ', valY.shape)
    print('test  X : ', testX.shape,  '\tY : ', testY.shape)

    np.save(data_dir+"/npset/"+season_mod+"_trainX",trainX)
    np.save(data_dir+"/npset/"+season_mod+"_trainY",trainY)
    np.save(data_dir+"/npset/"+season_mod+"_valX",valX)
    np.save(data_dir+"/npset/"+season_mod+"_valY",valY)
    np.save(data_dir+"/npset/"+season_mod+"_testX",testX)
    np.save(data_dir+"/npset/"+season_mod+"_testY",testY)

    #----------------------------------------------
    # Modeling
    #----------------------------------------------
    shape1 = trainX.shape[1]
    shape2 = trainX.shape[2]
    print("LSTM input shape : ", shape1, shape2)
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape1, shape2)))
    model.add(RepeatVector(hoursteps))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.summary()

    # model fit 
    for i in range(n_model):#0,5):#
        #keras.optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=RMSProp())
        hist = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(valX, valY))
        results = model.evaluate(testX, testY)
        #model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
        model.save(data_dir+'/model/model_'+season_mod+'_'+str(i)+'.h5')# # of feature=3,5,7,9,?,12,14,16,18

    #----------------------------------------------
    # train 과정 분석
    #----------------------------------------------
    print('result : ', results)

    #get test data
    X_test = np.load(data_dir+"/npset/"+season_mod+"_testX.npy")
    y_test = np.load(data_dir+"/npset/"+season_mod+"_testY.npy")

    print("X_test : ", X_test.shape)
    print("y_test : ", y_test.shape)

    n_dataset   = y_test.shape[0]
    acc_list    = []
    acc_model   = []
    model       = []
    result_list = {}

    for i in range(n_model):
        model.append(load_model(data_dir+'/model/model_'+season_mod+'_'+str(i)+'.h5'))
        acc_model.append(0)
        
    print("[ dataset ]")
    for i in range(n_dataset):
        #if(i in [0,5,13,14,24,25]): continue;
        y = sc_pow.inverse_transform(y_test[i:i+1,:,0])

        for m in range(n_model):
            #print("(model",m+1,")\t",end="")

            pred = model[m].predict([X_test[i:i+1]])
            pred[pred<0] = 0
            pred = pred[:,:,0]
            pred = sc_pow.inverse_transform(pred)
            pred = np.sum(pred)

            target      = round(np.sum(y), 2)
            error       = round(np.abs(target-pred), 2)
            error_rate  = np.min([round(error/target, 2),1])
            acc_rate    = round((1.0-error_rate)*100, 2)
            acc_list.append(acc_rate)
            acc_model[m] += acc_rate
                    
            #print("   pred: ",pred," | target: ",target," | error: ",error," | err rate: ",error_rate," | acc: ",acc_rate,sep="")
        #print("acc rate: ",np.mean(acc_list[-n_model:]),sep='')
        if(i%5==0): print(" ")
        print(round(np.mean(acc_list[-n_model:]), 2), " / ",sep='', end='')
        result_list[i] = round(np.mean(acc_list[-n_model:]), 2)
        
    print("\n----------------------------------------------")
    print("mean(acc rate): ",np.mean(acc_list),sep='')
    print("----------------------------------------------")
    print("[ model ]")
    for i in range(n_model):
        acc_model[i] = round(acc_model[i]/(n_dataset),2)
        print(acc_model[i])    
    print("type(result_list) : ", type(result_list))

    return result_list