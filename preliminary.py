import os
import numpy as np
import torch
from src.mutils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problems.tsp.problem_tsp import TSP
from torch.utils.data import DataLoader
import pandas as pd
from utils import load_model
from tqdm import tqdm
from utils.data_utils import save_dataset

dataset=TSP.make_dataset('data/tsp/tsp50_val_mg_seed2222_size10K.pkl')
dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
with open('data/tsp/tsp50_val_mg_seed2222_size10K.txt') as file:
    lines=file.readlines()
    gt=np.array([float(line) for line in lines])

modelpath='pretrained/tsp_50/'

def get_model_ratio(modelpath,return_all=False):
    model, _ = load_model(modelpath)
    model.eval()
    None
    pred2=get_costs_batch(model,dataloader)
    ratio2=(pred2[:len(gt)]-gt)/gt
    if return_all:
        return ratio2,pred2
    return ratio2

ratio3=get_model_ratio(modelpath)

# details of geneators refer to generate_data.py
opgs=[]
cdists="10 20 30 50 70 100".split()
for cdistidx,i in enumerate([2,3,4,6,8,11]):
    cdist=cdists[cdistidx]
    idx=range(i*833,(i+1)*833)
    r = 100*ratio3[idx]
    m = np.mean(r)
    dev = np.std(r)/np.sqrt(len(ratio3[idx]))
    m = round(m,2)
    dev = round(dev,2)
    opgs.append([cdist,m,dev])

df=pd.DataFrame(opgs,columns="cdist mean dev".split())
print(df)


