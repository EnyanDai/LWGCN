#%%
import time
import argparse
import numpy as np
import torch
from models.MLP import MLP
from models.GCN2 import GCN2
import warnings
warnings.filterwarnings('ignore')
from models.Bilevel import BilevelTrainer
from torch_geometric.datasets import Planetoid, WebKB
from dataset import WikipediaNetwork

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--dataset', type=str, default='Penn94', help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    from models.DIS_texas import DIS
    dataset = Planetoid(root='./data/',name=args.dataset,transform=transform)
    data = dataset[0].to(device)
    data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
    data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
    data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

if args.dataset in ["Texas", "Wisconsin",'Cornell']:
    from models.DIS_texas import DIS
    dataset = WebKB(root='./data/',name=args.dataset)
    data = dataset[0].to(device)

if args.dataset in ["crocodile", "squirrel",'chameleon']:
    from models.DIS_lp import DIS
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('./data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('./data/',name=args.dataset)
    data = dataset[0].to(device)

from utils import sys_normalized_adjacency,sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
import scipy.sparse as sp
adj = sys_normalized_adjacency(to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.x)))
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
#%%
from torch_geometric.utils import to_dense_adj
results = []
Bilevel_results = []
for i in range(data.train_mask.shape[1]):
    train_mask = data.train_mask[:,i]
    val_mask = data.val_mask[:,i]
    test_mask = data.test_mask[:,i]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    model = MLP(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=0.01,\
                weight_decay=5e-4,\
                device=device).to(device)

    #%%
    model.fit(data.x, data.y, train_mask, val_mask,train_iters=args.epochs)
    # print('======MLP=====')
    result = model.test(test_mask)
    # print(result)
    results.append(result)

    #%%
    pred = model(data.x).max(dim=1)[1]
    pred[train_mask] = data.y[train_mask].long()
    edge_label_wise = []
    
    for label in range(int(data.y.max())+1):
        mask = (pred==label)
        sub_adj = data.edge_index[:,mask[data.edge_index[1]]]
        if sub_adj.shape[1]<=0:
            continue
        dense_adj = to_scipy_sparse_matrix(sub_adj, num_nodes=len(data.x))
        rowsum = np.array(dense_adj.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        dense_adj = r_mat_inv.dot(dense_adj)
        dense_adj = sparse_mx_to_torch_sparse_tensor(dense_adj).to(device)
        edge_label_wise.append(dense_adj)
    
    if args.dataset in args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        layer = 2
    else:
        layer = args.layer

    dis_model = DIS(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.lr,\
                layer=layer,\
                weight_decay=args.weight_decay,\
                device=device, k=len(edge_label_wise)).to(device)
    
    if args.dataset in args.dataset in ['Cora', 'Citeseer', 'Pubmed',"Cornell"]:
        layer = args.layer
    else:
        layer = 2

    model = GCN2(data.x.shape[1], \
                nhidden=args.hidden,\
                nclass=int(data.y.max()+1),\
                nlayers=layer,\
                alpha=args.alpha,\
                lamda=args.lamda,\
                dropout=args.dropout, device=device).to(device)

    # model.fit(data.x, data.edge_index, data.y, args.lr, args.weight_decay, train_mask, val_mask, train_iters=500)
    bilevel = BilevelTrainer(
                    dis_model,\
                    model,\
                    lr=args.lr,\
                    weight_decay=args.weight_decay,\
                    device=device)
    bilevel.fit(data.x, edge_label_wise, adj, data.y, train_mask, val_mask,\
                inner_iters=1, train_iters=args.epochs,verbose=False)
    print('======{}th split====='.format(i+1))
    result = bilevel.test(test_mask)
    Bilevel_results.append(result)

print("Aveage: {:.4f}, std: {:.4f}".format(np.mean(Bilevel_results),np.std(Bilevel_results)))
# %%

# %%
