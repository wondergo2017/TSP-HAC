
import numpy as np
import torch
from src.mutils import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.options import get_options
from utils import torch_load_cpu, load_problem
import pandas as pd
set_random_seed_all(0)
opts=get_options('')
opts.no_progress_bar=True
problem = load_problem(opts.problem)
device=torch.device('cuda:0')
opts.device=device

model,dataset,dataloader,baseline,optimizer=init(pretrain=True,device=device,opts=opts)
size=100
res=[]
for eps in [0.1,0.5,1,5,10]:
    model.eval()
    set_decode_type(model, "greedy")
    ran_dataset = torch.FloatTensor(np.random.uniform(size=(size, 50, 2)))
    hard_data=get_hard_samples(model, ran_dataset, eps, batch_size=1024, device=device,baseline=baseline)
    ratio=get_gap(model,hard_data,device)
    ratio=ratio*100
    m = np.mean(ratio)
    dev = np.std(ratio)/np.sqrt(size)
    m = round(m,2)
    dev = round(dev,2)
    res.append([eps,m,dev])
df=pd.DataFrame(res,columns="eps mean dev".split())
print(df)

