import pickle
from tqdm import tqdm
from src.mutils import Solver
f=open(f'data/tsp/tsp50_val_mg_seed2222_size10K.txt','w')
data=pickle.load(open('data/tsp/tsp50_val_mg_seed2222_size10K.pkl','rb'))
for x in tqdm(data):  
    cost=Solver.gurobi(x)[0]
    f.write(str(cost))
    f.write('\n')
