from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.typing import EdgeType, NodeType, OptTensor
EPS = 1e-15
class randomWalk(torch.nn.Module):
    def __init__(
        self,
        dataset,
        edge_index_dict: Dict[EdgeType, Tensor],
        embedding_dim: int,
        metapath: List[EdgeType],
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes_dict: Optional[Dict[NodeType, int]] = None,
        sparse: bool = False,
    ):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                key = keys[0]
                N = int(edge_index[0].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

                key = keys[-1]
                N = int(edge_index[1].max() + 1)
                num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        adj_dict = {}
        for keys, edge_index in edge_index_dict.items():
            sizes = (num_nodes_dict[keys[0]], num_nodes_dict[keys[-1]])
            row, col = edge_index
            adj = SparseTensor(row=row, col=col, sparse_sizes=sizes)
            adj = adj.to('cpu')
            #每半个元路径有一个稀疏矩阵，用逗号隔开
            adj_dict[keys] = adj

        assert walk_length + 1 >= context_size
        if walk_length > len(metapath) and metapath[0][0] != metapath[-1][-1]:
            raise AttributeError(
                "The 'walk_length' is longer than the given 'metapath', but "
                "the 'metapath' does not denote a cycle")
        self.dataset = dataset
        self.adj_dict = adj_dict
        self.embedding_dim = embedding_dim
        self.metapath = metapath
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes_dict = num_nodes_dict

        types = set([x[0] for x in metapath]) | set([x[-1] for x in metapath])
        types = sorted(list(types))
        #types为元路径中的所有节点类型
        count = 0
        self.start, self.end = {}, {}
        for key in types:
            #计算每一种节点类型的开始与结束位置（相对），都是从0开始
            self.start[key] = count
            count += num_nodes_dict[key]
            self.end[key] = count
        #metapath[0][0]是元路径的开始节点类型
        #offset = 0
        offset = [self.start[metapath[0][0]]]
        offset += [self.start[keys[-1]] for keys in metapath
                   ] * int((walk_length / len(metapath)) + 1)
        offset = offset[:walk_length + 1]
        assert len(offset) == walk_length + 1
        self.offset = torch.tensor(offset)

        # + 1 denotes a dummy node used to link to for isolated nodes.
        # self.embedding = Embedding(count + 1, embedding_dim, sparse=sparse)
        # self.dummy_idx = dataset.num_nodes+1
        self.dummy_idx = self.dataset.num_nodes

        self.start_a_manage = 0
        self.start_b_produce = 3550
        self.start_c_environment = 7100
        self.start_d_holdCompany = self.start_c_environment + self.dataset['c_environment'].num_nodes
        self.start_e_staff = self.start_d_holdCompany + self.dataset['d_holdCompany'].num_nodes
        self.start_f_supplyCompany = self.start_e_staff + self.dataset['e_staff'].num_nodes
        self.start_g_sector = self.start_f_supplyCompany + self.dataset['f_supplyCompany'].num_nodes
        self.start_h_area = self.start_g_sector + self.dataset['g_sector'].num_nodes
        # print("节点数量节点数量节点数量节点数量节点数量节点数量节点数量节点数量节点数量节点数量")
        # print(self.start_d_holdCompany,self.start_e_staff,self.start_f_supplyCompany,self.start_g_sector,self.start_h_area)


    def loader(self, **kwargs):
        r"""Returns the data loader that creates both positive and negative
        random walks on the heterogeneous graph.

        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(range(self.num_nodes_dict[self.metapath[0][0]]),
                          collate_fn=self._sample, **kwargs)

    def _pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rws = [batch]
        #
        for i in range(self.walk_length):
            #len(self.metapath) = 半个元路径的数量
            keys = self.metapath[i % len(self.metapath)]
            adj = self.adj_dict[keys]
            batch = sample(adj, batch, num_neighbors=1,
                           dummy_idx=self.dummy_idx).view(-1)
            #在邻接矩阵中采样
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))
        rw[rw > self.dummy_idx] = self.dummy_idx

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        walks_return = torch.cat(walks, dim=0)

        nodetypes = set([x[0] for x in self.metapath]) | set([x[-1] for x in self.metapath])
        nodetypes = sorted(list(nodetypes))

        copy_walks_return = walks_return.clone()
        mask1 = (copy_walks_return!= self.dummy_idx) &(copy_walks_return<3550)
        mask2 = (copy_walks_return!= self.dummy_idx) &(copy_walks_return>=3550)

        if ('a_manage' in nodetypes) and ('d_holdCompany' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_a_manage
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_d_holdCompany
        elif ('a_manage' in nodetypes) and ('e_staff' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_a_manage
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_e_staff
        elif ('b_produce' in nodetypes) and ('f_supplyCompany' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_b_produce
            walks_return[mask2] = walks_return[mask2] - 3550 +  self.start_f_supplyCompany
        elif ('c_environment' in nodetypes) and ('g_sector' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_c_environment
            walks_return[mask2] = walks_return[mask2] - 3550 +  self.start_g_sector
        elif ('c_environment' in nodetypes) and ('h_area' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_c_environment
            walks_return[mask2] = walks_return[mask2] - 3550 +  self.start_h_area
        else:
            for i in range(100):
                print("error metapath type")
                print("the error at the file randomWalk,元路径变更，请更改代码")

        return walks_return

    def _neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rws = [batch]
        for i in range(self.walk_length):
            keys = self.metapath[i % len(self.metapath)]
            batch = torch.randint(0, self.num_nodes_dict[keys[-1]],
                                  (batch.size(0), ), dtype=torch.long)
            rws.append(batch)

        rw = torch.stack(rws, dim=-1)
        rw.add_(self.offset.view(1, -1))

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])

        walks_return = torch.cat(walks, dim=0)

        nodetypes = set([x[0] for x in self.metapath]) | set([x[-1] for x in self.metapath])
        nodetypes = sorted(list(nodetypes))
        copy_walks_return = walks_return.clone()
        mask1 = (copy_walks_return != self.dummy_idx) & (copy_walks_return < 3550)
        mask2 = (copy_walks_return != self.dummy_idx) & (copy_walks_return >= 3550)

        if ('a_manage' in nodetypes) and ('d_holdCompany' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_a_manage
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_d_holdCompany
        elif ('a_manage' in nodetypes) and ('e_staff' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_a_manage
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_e_staff
        elif ('b_produce' in nodetypes) and ('f_supplyCompany' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_b_produce
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_f_supplyCompany
        elif ('c_environment' in nodetypes) and ('g_sector' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_c_environment
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_g_sector
        elif ('c_environment' in nodetypes) and ('h_area' in nodetypes):
            walks_return[mask1] = walks_return[mask1] + self.start_c_environment
            walks_return[mask2] = walks_return[mask2] - 3550 + self.start_h_area
        else:
            for i in range(100):
                print("error metapath type")
                print("the error at the file randomWalk,元路径变更，请更改代码")

        return walks_return
    def _sample(self, batch: List[int]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
        return self._pos_sample(batch), self._neg_sample(batch)
        #  return batch


def sample(src: SparseTensor, subset: Tensor, num_neighbors: int,
           dummy_idx: int) -> Tensor:

    mask = subset < dummy_idx                                                    #生成非孤立的节点掩码

    rowcount = torch.zeros_like(subset)                                          #生成128维的0张量
    rowcount[mask] = src.storage.rowcount()[subset[mask]]                        #找到非孤立的节点（我的图的孤立节点似乎很多（好像是根据元路径来的））
    mask = mask & (rowcount > 0)
    offset = torch.zeros_like(subset)
    offset[mask] = src.storage.rowptr()[subset[mask]]

    rand = torch.rand((rowcount.size(0), num_neighbors), device=subset.device)
    rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(offset.view(-1, 1))

    col = src.storage.col()[rand]
    col[~mask] = dummy_idx
    return col
