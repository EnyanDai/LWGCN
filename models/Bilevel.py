import torch
import torch.nn.functional as F
import utils

import torch.optim as optim
from copy import deepcopy
class BilevelTrainer:

    def __init__(self, dis_model, pair_model, lr=0.01, weight_decay=5e-4,device=None):

        self.dis_model = dis_model
        self.GCN = pair_model
        self.weight = torch.tensor([0.0,0.0],requires_grad=True,device=device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device

    def fit(self, features, edge_label_wise, edge_index, labels, idx_train, idx_val=None, inner_iters=5 ,train_iters=200, verbose=False):

        self.edge_label_wise = edge_label_wise
        self.edge_index = edge_index
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.label_onehot = torch.eye(int(self.labels.max()+1), dtype = torch.float32, device=self.device)[self.labels]
        self._train_with_val(self.labels, idx_train, idx_val, inner_iters,train_iters, verbose)


    def _train_with_val(self, labels, idx_train, idx_val, inner_iters, train_iters, verbose):

        optimizer_dis = optim.Adam(self.dis_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_gcn = optim.Adam(self.GCN.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        optimizer_weight = optim.Adam([self.weight], lr=0.1)

        best_acc_val = 0

        for i in range(train_iters):

            for j in range(inner_iters):

                self.dis_model.train()
                self.GCN.train()

                optimizer_dis.zero_grad()
                optimizer_gcn.zero_grad()

                pred_ours = self.dis_model(self.features, self.edge_label_wise)
                pred_gcn = self.GCN(self.features, self.edge_index)

                norm_weight = F.softmax(self.weight.detach())

                final_pred = norm_weight[0]*(pred_ours) + norm_weight[1]*(pred_gcn)
                loss = -torch.sum(self.label_onehot[idx_train] * (final_pred[idx_train]),dim=1).mean()
                loss.backward()
                optimizer_dis.step()
                optimizer_gcn.step()

            self.dis_model.eval()
            self.GCN.eval()
            optimizer_weight.zero_grad()
            pred_ours = self.dis_model(self.features, self.edge_label_wise).detach()
            pred_gcn = self.GCN(self.features, self.edge_index).detach()

            norm_weight = F.softmax(self.weight)

            final_pred = norm_weight[0]*(pred_ours) + norm_weight[1]*(pred_gcn)
            loss = -torch.sum(self.label_onehot[idx_val] * (final_pred[idx_val]),dim=1).mean()
            loss.backward()
            optimizer_weight.step()

            acc_val = utils.accuracy(final_pred[idx_val], labels[idx_val])
            
            if verbose and i % 1 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))
                acc_ours = utils.accuracy(pred_ours[idx_val], labels[idx_val])
                acc_gcn = utils.accuracy(pred_gcn[idx_val], labels[idx_val])
                # print(self.weight)
                # print("acc_val: {:.4f}, ours: {:.4f}, gcn: {:.4f}".format(acc_val, acc_ours, acc_gcn))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = deepcopy(final_pred.detach())
                self.best_weight = deepcopy(self.weight.detach())


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        acc_test = utils.accuracy(self.output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "accuracy= {:.4f}".format(acc_test.item()))
        # print(float(F.softmax(self.best_weight.detach())[0]))
        return float(acc_test)