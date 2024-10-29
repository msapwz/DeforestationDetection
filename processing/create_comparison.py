import pandas as pd
import os
import numpy as np
import sys

path = sys.argv[1]#'../checkpoints'
dic = None

for i in os.listdir(path):
    data_info = i.split('_')
    file = os.path.join(path,i,'scores_dict.npy')
    if os.path.exists(file):

        data = np.load(file,allow_pickle=True)
        data = data.item()
        # print(data)

        if dic is None:
            columns = ['id','img_size','dataset','batch','learning_rate','optimizer','loss','epochs'] + list(data.keys())
            dic = {i:[] for i in columns}
        # print(data_info)
        for j in data.keys():
            print(j,data.keys())
            dic[j].append(data[j])
        dic['img_size'].append(data_info[3].split('-')[-2])
        dic['id'].append(i)
        dic['dataset'].append(data_info[3].split('-')[-1])
        dic['batch'].append(data_info[4][1:])
        dic['learning_rate'].append(data_info[5][2:])
        dic['optimizer'].append(data_info[6])
        dic['epochs'].append(data_info[9])
        dic['loss'].append(data_info[11])


df = pd.DataFrame(dic)
df.to_csv(os.path.join(path,"Results.csv"))
print(df)