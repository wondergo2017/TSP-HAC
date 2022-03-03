import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
import torch
from problems.tsp.tsp_baseline import run_insertion,nearest_neighbour
from functools import partial
from tqdm import tqdm
from utils import load_model
from problems.tsp.problem_tsp import TSP
import os
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from problems.tsp.problem_tsp import TSP
from torch.utils.data import DataLoader
import pandas as pd
from utils import load_model
from tqdm import tqdm
from utils.data_utils import save_dataset
from .options import get_options
from torch import optim
from utils import torch_load_cpu, load_problem
from .reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
import random
class Solver:
    @staticmethod
    def model(model,data):
        if len(data.shape)>2:
            return make_tour_batch(model,data)
        else:
            return make_tour(model,data)
    @staticmethod
    def gurobi(data):
        if len(data.shape)>2:
            return [solve_euclidian_tsp(x) for x in data]
        else:
            return solve_euclidian_tsp(data)

def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]

def make_tour_batch(model, batch):
    '''
    :params batch: Tensor 
    :return: list of [cost ,tour]
    '''
    model.eval()
    model.set_decode_type("greedy" )
    results=[]
    with torch.no_grad():
        batch_rep = 1
        iter_rep = 1
        sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
        batch_size = len(costs)
        ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
    if sequences is None:
        sequences = [None] * batch_size
        costs = [math.inf] * batch_size
    else:
        sequences, costs = get_best(
            sequences.cpu().numpy(), costs.cpu().numpy(),
            ids.cpu().numpy() if ids is not None else None,
            batch_size
        )
    for seq, cost in zip(sequences, costs):
        seq = seq.tolist()
        results.append([cost, seq])
    return results

def make_oracle(model, xy, temperature=1.0):
    model.eval()
    model.set_decode_type("greedy" )
    num_nodes = len(xy)
    
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
    
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            #assert np.allclose(p.sum().item(), 1)
        return p.numpy()
    
    return oracle
        

def make_tour(model,xy):
    '''
    xy : numpy 
    return : tour
    '''
    oracle = make_oracle(model, xy)
    sample = False
    tour = []
    tour_p = []
    while(len(tour) < len(xy)):
        p = oracle(tour)

        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)

    return tour

from gurobipy import *
def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: cost, tour
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    # You could use Python looping constructs and m.addVar() to create
    # these decision variables instead.  The following would be equivalent
    # to the preceding m.addVars() call...
    #
    # vars = tupledict()
    # for i,j in dist.keys():
    #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
    #                        name='e[%d,%d]'%(i,j))


    # Add degree-2 constraint

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))

    # Using Python looping constructs, the preceding would be...
    #
    # for i in range(n):
    #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)


    # Optimize model

    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour

def get_costs_batch(model,dataloader,device=torch.device('cuda:1')):
    '''
    get costs of dataloader 
    '''
    preds=[]
    model=model.to(device)
    with tqdm(dataloader) as bar:
        for data in bar:
            data=data.to(device)
            t=Solver.model(model,data)
            preds.extend([x[0] for x in t])
        preds=np.array(preds)
    return preds
    
def minmax(xy_):
    '''
    min max batch of graphs [b,n,2]
    '''
    xy_=(xy_-xy_.min(dim=1,keepdims=True)[0])/(xy_.max(dim=1,keepdims=True)[0]-xy_.min(dim=1,keepdims=True)[0])
    return xy_

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def mg(cdist=1000):
    '''
    GMM create one instance of TSP-50, using cdist
    '''
    nc=np.random.randint(3,7)
    nums=np.random.multinomial(50,np.ones(nc)/nc)
    xy=[]
    for num in nums:
        center=np.random.uniform(0,cdist,size=(1,2))
        nxy=np.random.multivariate_normal(mean=center.squeeze(),cov=np.eye(2,2),size=(num,))
        xy.extend(nxy)
    
    xy=np.array(xy)
    xy=MinMaxScaler().fit_transform(xy)
    return xy

def mg_batch(cdist,size):
    '''
    GMM create a batch size instance of TSP-50, using cdist
    '''
    xy=[]
    for i in range(size):
        xy.append(mg(cdist))
    return np.array(xy)

def generate_tsp_data_mg(size):
    '''
    formal test setting, generate GMM TSP-50 data (number size). every part size//12
    '''
    pern=size//12
    res=[]
    uni=np.random.uniform(size=(pern,50,2))
    res.append(uni)
    for cdist in [1,10,20,30,40,50,60,70,80,90,100]:
        res.append(mg_batch(cdist,pern))
    res=np.concatenate(res,axis=0)
    return res

from torch.utils.data import Dataset
class ConcatDataset(Dataset):
    '''
    concat a list of datasets
    '''
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length
    
from torch.utils.data import DataLoader
def get_hard_samples(model, data, eps=5, batch_size=1024,device='cpu',baseline=None):
    model.eval()
    set_decode_type(model, "greedy")
    def get_hard(model,data,eps):
        data = data.to(device)
        data.requires_grad_()
        cost, ll, pi = model(data, return_pi=True)
        if baseline is not None:
            with torch.no_grad():
                cost_b,_=baseline.model(data)
            cost,ll=model(data)
            delta = torch.autograd.grad(eps*((cost/cost_b)*ll).mean(),data)[0]
        else:
            # As dividend is viewed as constant, it can be omitted in gradient calculation. 
            delta = torch.autograd.grad(eps*(cost*ll).mean(), data)[0]
        ndata = data+delta
        ndata = minmax(ndata)
        return ndata.detach().cpu()
    dataloader = DataLoader(data, batch_size=batch_size)
    hard = torch.cat([get_hard(model, data, eps) for data in dataloader],dim=0)
    return hard

def get_gap(model,data,device):
    '''
    get gap for model on a batch of data
    '''
    data=data.cpu().numpy()
    hard_gt=[]
    for x in tqdm(data):  
        cost=Solver.gurobi(x)[0]
        hard_gt.append(cost)
    hard_gt=np.array(hard_gt)
    costs=Solver.model(model,torch.FloatTensor(data).to(device))
    costs=np.array([c[0] for c in costs])
    ratio=(costs-hard_gt)/hard_gt
    info=[ratio.mean(),costs.mean(),hard_gt.mean()]
    return ratio



from nets.attention_model import AttentionModel
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
def init(pretrain=True,device=torch.device('cpu'),opts=None):
    '''
    init a TSP-50 model. using uniform 10k data for baseline
    '''
    problem = load_problem(opts.problem)
    # model
    if pretrain:
        model, _ = load_model('pretrained/tsp_50/')
    else:
        model_class = {
            'attention': AttentionModel,
            'pointer': PointerNetwork
        }.get(opts.model, None)
        model = model_class(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping,
            checkpoint_encoder=opts.checkpoint_encoder,
            shrink_size=opts.shrink_size
        ).to(opts.device)
    model=model.to(device)
    model.set_decode_type("greedy" )
    model.eval()  # Put in evaluation mode to not track gradients

    # dataset
    dataset=TSP.make_dataset('data/tsp/tsp50_train_seed1111_size10K.pkl')
    dataloader=DataLoader(dataset,batch_size=1024)
    # baseline
    
    baseline = RolloutBaseline(model, problem, opts,dataset=dataset)
    # optim
    optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0
                else []
            )
        )
    return model,dataset,dataloader,baseline,optimizer

from copy import deepcopy
from nets.attention_model import set_decode_type
from utils import move_to
from .train import *


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def greedy_batch_seq(model, batch):
    model.eval()
    model.set_decode_type("greedy" )
    sequences, costs = model.sample_many(batch, batch_rep=1, iter_rep=1)
    return sequences

def plot_one(ax,model,data):
    '''
    plot one solution of data(shape [1,1]) on ax
    '''
    xy=data[0].detach().cpu().numpy()
    tour=greedy_batch_seq(model,data).cpu().numpy()[0]
    gtc,tour2=solve_euclidian_tsp(xy)
    plot_tsp(xy, tour, ax,tour2,gtc)

def plot_tsp(xy, tour, ax1,tour2=None,cost2=None):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()
#     print(d)
    # Scatter nodes
    ax1.scatter(xs, ys, s=40, color='blue')
    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=40, color='blue')
    
    
    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
        alpha=0.5
    )
    
    if tour2 is not None:
        xs2,ys2,dx2,dy2,c=getdelta(xy,tour2)
        qv = ax1.quiver(
            xs2, ys2, dx2, dy2,
            scale_units='xy',
            angles='xy',
            scale=1,
            color='green',
            alpha=0.5
        )

        ax1.set_title('{} nodes, optimal cost {:.1f}, gap {:.1f}%'.format(len(tour), cost2,100*(lengths[-1]-cost2)/cost2))
    else:
        ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))

def getdelta(xy,tour):
    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = sum(d)
    return xs,ys,dx,dy,lengths