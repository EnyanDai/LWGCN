#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,JumpingKnowledge
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class DISConv(nn.Module):
    def __init__(self):

        super(DISConv, self).__init__()

    def forward(self, x, edge_label_wise):
        h = []
        # x = self.linear(x)
        for edge_index in edge_label_wise:
            if edge_index.is_sparse:
                h.append(torch.spmm(edge_index, x))
            else:
                h.append(edge_index@x)
        h.append(x)
        
        h = torch.cat(h, dim=1)
        return h



class DIS(nn.Module):

    def __init__(self, nfeat, nhid, nclass, k, layer=2,dropout=0.5, lr=0.01, weight_decay=5e-4,device=None):

        super(DIS, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        #k = len(edge_label_wise)
        # self.jump = JumpingKnowledge(mode="cat")
        self.convs = nn.ModuleList()
        self.convs.append(DISConv())
        self.convs.append(nn.Sequential(nn.Linear((k+1)*nfeat,nhid),nn.ReLU()))
        for _ in range(layer-2):
            self.convs.append(DISConv())
            self.convs.append(nn.Sequential(nn.Linear((k+1)*nhid,nhid),nn.ReLU()))
        self.convs.append(DISConv())
        self.convs.append(nn.Linear((k+1)*nhid,nclass))
        # self.cls = nn.Sequential(nn.Linear(nhid,nclass))

        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_label_wise = None
        self.features = None

    def forward(self, x, edge_label_wise):
        # h = []
        for i in range(0,len(self.convs),2):
            x = self.convs[i](x,edge_label_wise)
            x = self.convs[i+1](x)

        y = x

        return F.log_softmax(y,dim=1)

    def get_h(self, x, edge_label_wise):
       
        for i in range(0,len(self.convs)-2,2):
            x = self.convs[i](x,edge_label_wise)
            x = self.convs[i+1](x)
        
        return x

    def fit(self, features, edge_label_wise, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """


        self.edge_label_wise = edge_label_wise
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)

        self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)


    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_label_wise)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_label_wise)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}, loss_val: {:.4f}".format(acc_val,loss_val))
            if acc_val > best_acc_val :
                
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

            # if loss_val < best_loss_val:
            #     best_loss_val = loss_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.forward(self.features, self.edge_label_wise)
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)