from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.data import DataLoader,Batch
from torch_geometric.nn.glob.glob import global_max_pool

import torch
import torch.nn.functional as F
import torch.nn as nn

class CCRGNN(nn.Module):
    def __init__(self,input_channel,out_channel):
        super(CCRGNN,self).__init__()
        hid = 32
        self.gcn1 = GATConv(input_channel, 8)
        self.gcn2 = GATConv(8, 64)
        self.gcn3 = GATConv(64, 32)
        self.gcn4 = GATConv(32, out_channel)

        self.lin1 = nn.Linear(4560,1024)
        self.lin2 = nn.Linear(1024,128)
        self.lin3 = nn.Linear(128,9)
        # self.reset_parameters()
    def reset_parameters(self):
        glorot(self.labelemb)
    def forward(self,batch):
        batch_size = batch.batch.max()+1
        x = batch.x.cuda()
        res = x.reshape(batch_size,-1)
        out = global_max_pool(x.reshape(-1,1),batch.batch.cuda())
        edge_index = batch.edge_index.cuda()

        x = F.relu(self.gcn1(x.reshape(-1,1),edge_index))
        res1 = x.reshape(batch_size,-1)
        out1 = global_max_pool(x,batch.batch.cuda())

        x = F.relu(self.gcn2(x, edge_index))
        res2 = x.reshape(batch_size,-1)
        out2 = global_max_pool(x,batch.batch.cuda())

        x = F.relu(self.gcn3(x, edge_index))
        res3 = x.reshape(batch_size,-1)
        out3 = global_max_pool(x,batch.batch.cuda())

        x = F.relu(self.gcn4(x, edge_index))
        res4 = x.reshape(batch_size,-1)
        out4 = global_max_pool(x,batch.batch.cuda())


        # print(res.shape)
        # print(res1.shape)
        # print(res2.shape)
        # print(res3.shape)
        # print(res4.shape)

        # print(out.shape)
        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # print(out4.shape)
        finalout = torch.cat([res,res1,res2,res3,res4,out,out1,out2,out3,out4],1)
        finalout = F.relu(self.lin1(finalout))
        finalout = F.relu(self.lin2(finalout))
        finalout = self.lin3(finalout)

        return finalout