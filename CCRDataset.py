import torch
import pickle
import numpy as np
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.utils import k_hop_subgraph,subgraph,contains_isolated_nodes
import networkx as nx 
import matplotlib.pyplot as plt

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

class CCRDataset(InMemoryDataset):
    def __init__(self, root,transform=None, pre_transform=None):
        super(CCRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.npz']

    @property
    def processed_file_names(self):
        return ['CCRGraphs.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        datafile = self.raw_dir + "/" +self.raw_file_names[0]
        # train 2557 test 640
        data = np.load(datafile)
        train_x,test_x,train_y,test_y = data["train_x"],data["test_x"],data["train_y"],data["test_y"]    
        data = np.concatenate((train_x,test_x))
        label = np.concatenate([train_y,test_y])
        for i in range(data.shape[0]):
            print(i)
            d = data[i].reshape(-1,1)
            product = np.exp(d.dot(d.T))
            product = noramlization(product)
            idx = self.product2edgeindex(product)
            data_list.append(Data(edge_index = torch.Tensor(idx).long(), x = torch.Tensor(data[i]), y = torch.Tensor([label[i]]).long()))
            # print(Data(edge_index = torch.Tensor(idx).long(), x = torch.Tensor(data[i]), y = torch.Tensor(label[i])))
            # exit()
        # print(data.shape)
        # print(label.shape)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    def product2edgeindex(self,product):
        value = 0.9
        while True:
            idx = np.where(product>value)
            # index =
            res = contains_isolated_nodes(torch.Tensor(idx).long(), num_nodes=39)
            G=nx.Graph()
            G.add_edges_from(np.array(idx).T.tolist())
            x = self.getncom(G)
            if x==1:
                return  np.unique(np.array([np.hstack([idx[0],idx[1]]),np.hstack([idx[1],idx[0]])],dtype=np.int),axis=1).astype(np.int)
            value -= 0.1
    def getncom(self,G):
        c = 0
        for i in nx.connected_components(G):
            c += 1
        return c
