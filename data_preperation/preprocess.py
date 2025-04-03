import torch 
import pandas as pd 
import numpy as np
from tqdm import tqdm
import pickle as pkl
config={
    "data":"/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/data/ADANIPORTS_minute.csv",
    "saved_data":"/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/saved_data"
}

class DataPreprocess():

    def __init__(self,config,time_window,sliding_window,save):

        self.config=config
        data_path=self.config['data']

        self.raw_data=pd.read_csv(open(data_path,'r'))

        self.time_window=time_window
        self.sliding_window=sliding_window

        self.Final_Data_x=[]
        self.Final_Data_y=[]
        self.save=save

    def get_data(self,load_from):

        if load_from!=None:
            data=pkl.load(open(load_from,'rb'))
            return data

        self.raw_data[['date','time']]=self.raw_data['date'].str.split(expand=True)

        group=self.raw_data.groupby('date').groups   
        for i in tqdm(group):
            day=self.raw_data[group[i][0]:group[i][-1]]
            converted=np.array(self.converting_timestamp(day))
            self.sliding_data(converted)

        self.Final_Data_x=torch.tensor(np.array(self.Final_Data_x),dtype=torch.float32)
        self.Final_Data_y=torch.tensor(np.array(self.Final_Data_y),dtype=torch.float32)
        
        # print(self.raw_data[0:15])
        # print(self.Final_Data_x[:6])
        # print(self.Final_Data_y[:6])

        if self.save==True:
            self.save_data()
        return (self.Final_Data_x,self.Final_Data_y)
    
    def converting_timestamp(self,data):
        day_data=[]
        count=0
        for i in range(0,len(data)//self.time_window):

            x=data.iloc[count:count+self.time_window]
            count+=self.time_window
            day_data.append([x['open'].iloc[0],x['high'].max(),x['low'].min(),x['close'].iloc[-1],x['volume'].sum()])
        return day_data

    def sliding_data(self,x):

        for i in range(0 ,len(x)-self.sliding_window):
            
            # 3rd column is close column
            self.Final_Data_y.append(x[i+self.sliding_window][3])
            self.Final_Data_x.append(np.hstack(list(x[i:i+self.sliding_window])))

    def save_data(self):

        path=self.config['saved_data']+"/data_"+str(self.time_window)+"_"+str(self.sliding_window)+".pkl"
        print("data is saved in path "+path)
        pkl.dump((self.Final_Data_x,self.Final_Data_y),open(path,'wb'))

obj=DataPreprocess(config,3,3,True)

data=obj.get_data(load_from="/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/saved_data/data_3_3.pkl")

