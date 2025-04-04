import torch 
import pandas as pd 
import numpy as np
from tqdm import tqdm
import pickle as pkl
from torch.utils.data import Dataset,DataLoader,random_split

class DataPreprocess():

    def __init__(self,config,size,save):

        self.config=config
        data_path=self.config['data']

        self.raw_data=pd.read_csv(open(data_path,'r'))

        self.size=size
        self.save=save

    def get_data(self):

        # if load_from!=None:
        #     self.Final_Data_x,self.Final_Data_y=pkl.load(open(load_from,'rb'))
        #     return self.Final_Data_x,self.Final_Data_y

        self.raw_data[['date','time']]=self.raw_data['date'].str.split(expand=True)

        group=self.raw_data.groupby('date').groups

        self.raw_data=self.raw_data[['open','high','low','close','volume']]

        self.data_x=[]   
        self.data_y=[]   
        for i in tqdm(group):
            day=self.raw_data[group[i][0]:group[i][-1]]
            if self.size**2<=len(day):
                op= self.make_grid(day)
                self.data_x.append(op[0])
                self.data_y.append(np.array(op[1]).reshape(-1,1))

        self.data_x=np.vstack(self.data_x)
        self.data_y=np.vstack(self.data_y)

        return self.data_x,self.data_y


    def make_grid(self,x):
        data_y=[]
        s=self.size**2
        x_temp=x['close'].to_numpy()
        x=np.array(x)
        data_x=[]
        for i in range(len(x)-s-1):
            temp=[]
            for j in range(5):
                temp.append(x[i:s+i,j].reshape((self.size,self.size)))
            data_y.append(x_temp[s+i])
            data_x.append(temp)
        return data_x,data_y

    def laoded_data(self,x,y,batch_size):
        
        dataset=MyDataset(x,y)

        data=random_split(dataset,[0.8,.2])
        train_data=DataLoader(data[0],batch_size)
        test_data=DataLoader(data[1])

        return train_data,test_data


    
class MyDataset(Dataset):
    def __init__(self,x,y):

        self.x=torch.tensor(x,dtype=torch.float32,requires_grad=True)
        self.y=torch.tensor(y,dtype=torch.float32,requires_grad=True)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):

        return self.x[index],self.y[index]
    