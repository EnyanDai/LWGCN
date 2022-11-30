#%%
import argparse
import numpy as np
import torch
from models.MLP import MLP
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.datasets import Planetoid, WebKB

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.005,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=2)
parser.add_argument('--dataset', type=str, default='Cornell', help='Random seed.')
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
# from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WebKB,Planetoid
transform = T.Compose([T.NormalizeFeatures()])

if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
    from models.DIS_texas import DIS
    dataset = Planetoid(root='./data/',name=args.dataset,transform=transform)
    data = dataset[0].to(device)
    data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
    data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
    data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

if args.dataset in ["Texas", "Wisconsin","Cornell"]:
    from models.DIS_texas import DIS
    dataset = WebKB(root='./data/',name=args.dataset)
    data = dataset[0].to(device)

if args.dataset in ["crocodile", "squirrel",'chameleon']:
    from models.DIS_lp import DIS
    from dataset import WikipediaNetwork
    if args.dataset=="crocodile":
        dataset = WikipediaNetwork('./data/',name=args.dataset,geom_gcn_preprocess=False)
    else:
        dataset = WikipediaNetwork('./data/',name=args.dataset)
    data = dataset[0].to(device)
#%%

from utils import sparse_mx_to_torch_sparse_tensor
from torch_geometric.utils import to_scipy_sparse_matrix

results = []
dis_results = []
for i in range(data.train_mask.shape[1]):
    train_mask = data.train_mask[:,i]
    val_mask = data.val_mask[:,i]
    test_mask = data.test_mask[:,i]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #%%
    if args.dataset == "arxiv-year":
        weight_deacy = 0.0
    else:
        weight_deacy = 5e-4

    model = MLP(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=0.01,\
                weight_decay=weight_deacy,\
                device=device).to(device)

    #%%
    model.fit(data.x, data.y, train_mask, val_mask,train_iters=args.epochs)
    print('======MLP=====')
    result = model.test(test_mask)
    print(result)
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

    dis_model = DIS(nfeat=data.x.shape[1],\
                    nhid=args.hidden,\
                    nclass= int(data.y.max()+1),\
                    dropout=args.dropout,\
                    lr=args.lr,\
                    layer=args.layer,\
                    weight_decay=args.weight_decay,\
                    device=device, k=len(edge_label_wise)).to(device)
    dis_model.fit(data.x, edge_label_wise, data.y, train_mask, val_mask,train_iters=args.epochs)
    print('======{}th split====='.format(i+1))
    dis_result = dis_model.test(test_mask)
    print("Test set results: accuracy= {:.4f}".format(dis_result))
    dis_results.append(dis_result)

# print(np.mean(results),np.std(results))
print("Aveage: {:.4f}, std: {:.4f}".format(np.mean(dis_results),np.std(dis_results)))

# %%