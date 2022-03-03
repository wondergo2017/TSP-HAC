import os
import numpy as np
import torch
from src.mutils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.options import get_options
from torch import optim
from utils import torch_load_cpu, load_problem
opts=get_options('')
opts.no_progress_bar=True
problem = load_problem(opts.problem)
device=torch.device('cuda:0')
opts.device=device
model,dataset,dataloader,baseline,optimizer=init(pretrain=True,device=device,opts=opts)
dataset=dataset[:100]
epss=[1,3,5]

def get_delta(eps=1):
    data=torch.stack(dataset)
    model.eval()
    baseline.model.eval()
    model.set_decode_type('greedy')
    baseline.model.set_decode_type('greedy')
    data.requires_grad_();
    data=data.to(device)
    with torch.no_grad():
        cost_b,_,_=baseline.model(data,return_pi=True)
    cost,ll,pi=model(data,return_pi=True)
    delta=torch.autograd.grad(eps*((cost/cost_b)*ll).mean(),data)[0]
    return delta
deltas=[]
for eps in epss:
    deltas.append(get_delta(eps))

start=25
num=6
cnum=4
fig,axes=plt.subplots(num,cnum,figsize=(cnum*5,num*5),dpi=300)

data=torch.stack(dataset)
data=data.to(device)
for i,idx in enumerate(range(6)):
    ax=axes[i][0]
    plot_one(ax,model,data[idx:idx+1])
    for j in range(cnum-1):
        delta=deltas[j]
        xy_=data+delta
        xy_=(xy_-xy_.min(dim=1,keepdims=True)[0])/(xy_.max(dim=1,keepdims=True)[0]-xy_.min(dim=1,keepdims=True)[0])
        ax2=axes[i][j+1]
        plot_one(ax2,model,xy_[idx:idx+1])
for i in range(4):
    axes[num-1][i].set_xlabel((['original']+[r'$\eta={}$'.format(t) for t in[1,3,5]])[i],fontsize=20)
plt.tight_layout()
os.makedirs('results',exist_ok=True)
plt.savefig('results/hag-case.png')