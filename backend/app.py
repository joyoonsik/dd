# python app.py
from flask import Flask,request,json,jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from library import pvBackend

app = Flask(__name__)
CORS(app)

@app.route('/calPC',methods=['GET', 'POST'])
def dataTransformation():
    params  = json.loads(request.get_data(), encoding='utf-8')
    datas   = params['datas']
    data    = np.array(datas)
    DF      = pd.DataFrame(data=data[1:,0:], columns=data[0,0:])  # 1st row as the column names
    ####################윤식추가###################
    data = DF.copy()
    data['ymdhms'] = data['ymdhms'].astype(str) 
    tmp = data[['ymdhms']]
    tmp.columns = ['key']
    data = pd.concat([data,tmp],axis=1)
    data['key'] = data['key'].str[4:8]
    data['ymdhms'] = data['ymdhms'].str[8:]
    data = data.sort_values(by=['key','ymdhms'], ascending=True) 
    data = data.reset_index(drop=True)
    data['ymdhms'] = data['ymdhms'].astype(int)
    FinalDF = pd.DataFrame(columns=data.columns)
    data['key'] = data['key'].astype(str)
    for i in range(12):
        tmp = data[data['key'].str[0:2].astype(int) == i+1]
        tmp['ymdhms'] = tmp['ymdhms'] + i*24
        FinalDF = pd.concat([FinalDF,tmp])
    LineChartJson = FinalDF.to_json(orient="records")
    ####################윤식추가###################
    cols    = DF.columns
    DF[cols]= DF[cols].apply(pd.to_numeric, errors='coerce')
    
    scaler      = MinMaxScaler(feature_range = (0, 1))
    scaled_total= scaler.fit_transform(DF.values)
    df_total    = pd.DataFrame(scaled_total, columns=DF.columns, index=list(DF.index.values))

    corr = df_total.corr('pearson')
    corr = corr.iloc[0:1,:]
    
    print(type(corr))
    print(type(LineChartJson))
    print("*corr : \n", corr)

    result = corr.to_json(orient="records")
    print(type(result))
    ####################윤식추가###################
    df_row = DF.shape[0]
    df_col = DF.shape[1]
    returnVal = { 
        'row' : df_row,
        'col' : df_col,
        'result' : result,
        'LineChartJson' : LineChartJson 
    }
    ####################윤식추가###################
    return json.dumps(returnVal)

@app.route('/getTD',methods=['GET', 'POST'])
def getTransformedData():
    params          = json.loads(request.get_data(), encoding='utf-8')
    transformMode   = params['mode']
    dataSet         = params['data']
    dataSet         = pd.DataFrame(np.array(dataSet[1:]), columns=dataSet[0])
    bins            = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    origin_histo    = {}
    transf_histo    = {}
    
    transformedData = pvBackend.transform(transformMode, dataSet)

    #--------------- array version
    origin_histo_arr = []
    transf_histo_arr = []
    for col in transformedData.columns:
        origin_histo[col], temp_bins = list(np.histogram(dataSet[col], bins))   
        transf_histo[col], temp_bins = list(np.histogram(transformedData[col], bins))   
        print(transf_histo[col])
        origin_histo_arr.append(origin_histo[col])
        transf_histo_arr.append(transf_histo[col])

    origin_df = pd.DataFrame(data=origin_histo_arr, columns=bins[1:])
    transf_df = pd.DataFrame(data=transf_histo_arr, columns=bins[1:])

    print("array version : ")
    print(origin_df)
    print(transf_df)

    #--------------- dataframe version
    for col in transformedData.columns:
        origin_histo[col], temp_bins = list(np.histogram(dataSet[col], bins))   
        transf_histo[col], temp_bins = list(np.histogram(transformedData[col], bins))   
        
    print("dataframe version : ")
    df = pd.DataFrame(data=transf_histo)
    print("df : \n", df)

    #--------------- array/dataframe test
    
    origin_result = origin_df.to_json(orient="records")
    transf_result = transf_df.to_json(orient="records")
    #return json.dumps({"origin": origin_result, "transf": transf_result})
    return json.dumps(transf_result)

@app.route('/runDL',methods=['GET', 'POST'])
def runDeepLearning():
    print("app.py > runDeepLearning : " )
    dlData = pvBackend.trainDlModel();
    print("deepLearnedData : ", type(dlData), " : ", dlData)
    
    dlData = pd.DataFrame.from_dict(dlData, orient='index') #transformedData['sunshine']
    result = dlData.to_json(orient="records")
    return json.dumps(result)

if __name__ == '__main__':
    app.run(debug=True)
