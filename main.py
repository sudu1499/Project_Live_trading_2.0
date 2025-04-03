from data_preperation.preprocess import DataPreprocess
import json


config=json.load(open("/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/config.json"))

obj=DataPreprocess(config,3,3,True)

data=obj.get_data(load_from="/mnt/6A8CB2D58CB29AD1/Project_Live_trading_2.0/saved_data/data_3_3.pkl")