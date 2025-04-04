from data_preperation.preprocess import DataPreprocess
import json
from model.pytorch_model import CnnLstm
import torch
from torch import nn

config=json.load(open("/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/config.json"))

obj=DataPreprocess(config,10,True)

data=obj.get_data(load_from="/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/saved_data/data_3_3.pkl")

in_channels=5
out_channels=10
kernel_size=3
stride=1


model=CnnLstm(in_channels,out_channels,kernel_size,stride)

model(torch.tensor(data[0][0],dtype=torch.float32))

