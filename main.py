from argparse import ArgumentParser
from src.train import *
from utils import move_to
import numpy as np
import torch
from src.mutils import *
import numpy as np
from problems.tsp.problem_tsp import TSP
from torch.utils.data import DataLoader
from src.options import get_options
from utils import torch_load_cpu, load_problem
from tensorboardX import SummaryWriter

opts = get_options('')
set_random_seed_all(0, deterministic=True)
opts.no_progress_bar = True
problem = load_problem(opts.problem)
device = torch.device('cuda:0')
opts.device = device

def train_batch(
        model,
        optimizer,
        baseline,
        batch,
        epoch,
        step,
        writer=None
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, device)
    bl_val = move_to(bl_val, device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    loss = (cost - bl_val) * log_likelihood

    if opts.reweight == 1:
        loss = ((cost-bl_val) * log_likelihood)
        w = ((cost/bl_val) * log_likelihood).detach()
        t = torch.FloatTensor([20-(epoch % 20)]).to(loss.device)
        w = torch.tanh(w)
        w /= t
        w = torch.nn.functional.softmax(w, dim=0)
        reinforce_loss = (w*loss).sum()
    else:
        reinforce_loss = (loss).mean()

    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    return cost.mean().item(), loss.mean().item()

def train_epoch(model, optimizer, baseline, opts,train_dataset=None,epoch=1,step=1,writer=None):
    training_dataset = baseline.wrap_dataset(train_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1,shuffle=True,drop_last=True)
            
    model.train()
    set_decode_type(model, "sampling")
    bar=tqdm(training_dataloader, disable=opts.no_progress_bar)
    costs=[]
    losses=[]
    hards=[]
    for batch_id, batch in enumerate(bar):
        cost,loss=train_batch(
            model,
            optimizer,
            baseline,
            batch,
            epoch,opts.batch_size*epoch+batch_id
            ,writer
        )
        bar.set_postfix(cost=cost)
        costs.append(cost)
        losses.append(loss)
    baseline.epoch_callback(model, epoch,dataset=train_dataset)
    return np.mean(costs),np.mean(losses)

def get_hard_data1(model, dataset=None, size=10000, eps=5, baseline = None):
    # random generate and output hard
    ran_dataset = torch.FloatTensor(np.random.uniform(size=(size, 50, 2)))
    ran_dataset = get_hard_samples(model, ran_dataset, eps, batch_size=1024, device=device, baseline=baseline)
    return ran_dataset

def get_hard_data2(model, dataset=None, size=10000, *args, **kwargs):
    # random generate
    ran_dataset = torch.FloatTensor(np.random.uniform(size=(size, 50, 2)))
    return ran_dataset

val_dataset = TSP.make_dataset('data/tsp/tsp50_val_mg_seed2222_size10K.pkl')
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
with open('data/tsp/tsp50_val_mg_seed2222_size10K.txt') as file:
    lines = file.readlines()
    gt = np.array([float(line) for line in lines])
    
def get_ratio(model, val_dataset, opts):
    costs = validate(model, val_dataset, opts, True)[-1].numpy()
    ratio = (costs-gt)/gt
    return [ratio.mean(), ratio.max(), costs.mean(), costs.max()]

def test(size=10000, eps=5, use_hard=0, instep=5, reweight=0, epochs=10, moving=0, it=None,log_dir='logs/'):
    opts.reweight=reweight
    get_hard = get_hard_data1 if use_hard else get_hard_data2
    model, dataset, dataloader, baseline, optimizer = init(True,device,opts)

    gaps = []
    def record(x, epoch):
        gaps.append(x[0])

    x = np.array(get_ratio(model, val_dataset, opts))
    record(x,0)
    basedata = dataset[:opts.val_size]
    ran_dataset = get_hard(model, dataset, size=size, eps=eps,baseline=baseline)
    train_d = ConcatDataset([basedata, ran_dataset])
    baseline._update_model(model,0,train_d)
    for epoch in range(epochs):
        ran_dataset = get_hard(model, dataset, size=size, eps=eps,baseline=baseline)
        train_d = ConcatDataset([basedata, ran_dataset])
        res = []
        for t in range(instep):
            step = epoch*instep+t
            cost, loss = train_epoch(
                model, optimizer, baseline, opts, train_d, epoch=epoch, step=step)
            x = np.array(get_ratio(model, val_dataset, opts))
            res.append(x)
        x = np.mean(res, axis=0)

        if epoch % 10 == 9:
            record(x, epoch)

    return gaps
    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_type', type=str, choices='uniform hardness-adaptive'.split())
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='logs/tmp')
    
    args = parser.parse_args()
    train_type=args.train_type
    if train_type == 'uniform':
        reweight = 0
        use_hard = 0
    elif train_type == 'hardness-adaptive':
        reweight = 1
        use_hard = 1
    else:
        raise NotImplementedError(f'')

    instep = 1
    epochs = 30
    eps = 5
    iters = args.iters

    res=[]
    for it in range(iters):
        gaps = test(use_hard=use_hard, instep=instep, epochs=epochs, reweight=reweight, eps=eps, it=it, log_dir= args.log_dir)
        res.append(gaps)
        print(res)

    res = np.array(res)*100
    m = np.mean(res,axis=0)
    dev = np.std(res,axis=0)/np.sqrt(len(res))
    m = [round(x,2) for x in m]
    dev = [round(x,2) for x in dev]
    
    df = []
    for i in range(len(m)):
        epoch=[0,10,20,30][i]
        df.append([epoch,m[i],dev[i]])
    df = pd.DataFrame(df,columns='epoch mean dev'.split())
    print(f'Train-type {train_type}')
    print(df)
