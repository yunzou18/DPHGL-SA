from torch import nn
from torch_geometric.nn import HANConv
from typing import Dict, List, Union
from torch.nn import Embedding
import torch
class HAN(nn.Module):
    def __init__(self, in_channels:Union[int, Dict[str, int]], hidden_channels, out_channels):  
        super(HAN, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = HANConv(in_channels, hidden_channels, graph.metadata(), heads=4)
        self.conv2 = HANConv(hidden_channels, out_channels, graph.metadata(), heads=4)


    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        print(x_dict)
        print(edge_index_dict)
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = x['a_manage']
        y = x['b_produce']
        z = x['c_environment']


        output = torch.cat(x,y,z)


        return output ,self.embedding.weight
