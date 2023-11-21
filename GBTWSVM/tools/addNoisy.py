import pandas as pd
import collections
import copy
import warnings
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

def recreat_data(data,pre):
    """
    :return:
    """
    """
    df=pd.read_csv(url,header=None)
    data=df.values
    """
    numSamples,numAttribute=data.shape
    samNums=(int)(numSamples*pre)
    df=pd.DataFrame(data)
    k_list=list(collections.Counter(data[:,numAttribute-1]).keys())
    v_list=list(collections.Counter(data[:,numAttribute-1]).values())
    # print(k_list)
    dff={}
    for i in range(len(k_list)):
        tag=k_list[i]#标签类别
        dff[i]=df[df[numAttribute-1]==int(tag)].reset_index()
        samNumsi=int(samNums*(v_list[i]/numSamples))
        temp_k=copy.deepcopy(k_list)
        temp_v=copy.deepcopy(v_list)
        temp_k.pop(i)
        temp_v.pop(i)
        k=0
        for j in range(len(temp_k)):
            samNumsij=int(samNumsi*(temp_v[j]/(sum(temp_v))))
            # print("sam",samNumsij)
            for l in range(k,samNumsij):
                dff[i].loc[l, numAttribute-1] = int(temp_k[j])
                k+=1
    new=dff[0]
    for i in range(len(k_list)-1):
        new=pd.concat([new,dff[i+1]])
    new = shuffle(new).reset_index().drop(['index', 'level_0'], axis=1)
    return new.values
