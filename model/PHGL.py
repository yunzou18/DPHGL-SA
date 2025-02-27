from torch import nn
from torch_geometric.nn import HANConv
from typing import Dict, List, Union
from torch.nn import Embedding
import torch
class HAN(nn.Module):
    def __init__(self, in_channels:Union[int, Dict[str, int]],out_channels, hidden_channels,graph,orther_count):  
        super(HAN, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = HANConv(in_channels, hidden_channels, graph.metadata(), heads=4)
        self.embedding = Embedding(orther_count + 1, hidden_channels, sparse=False)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)


        # a = x['a_manage']
        # b = x['b_produce']
        # c = x['c_environment']
        a = x['abc_stock']

        # output = torch.cat((a,b,c),dim=0)
        output =  a


        return output ,self.embedding.weight 
