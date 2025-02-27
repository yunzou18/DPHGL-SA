import pandas as pd
import torch
import torch_geometric.transforms as T
from CreateDataset import MyHeteroDataset
import torch.nn as nn
from Model.randomWalk import randomWalk
from Model.abcdefg import notdivision_randomWalk
from Model.HANLSTM import HANLSTM

# step 1
root = './GraphDataset-notonehot-notdivision/'
root2 = './GraphDataset-not-onehot-division/'
root3 = './GraphDataset/'
dataset = MyHeteroDataset(root=root)
# dataset.process()
data = dataset.get(23)
# count = orther['d_holdCompany'].num_nodes + orther['e_staff'].num_nodes + orther['f_supplyCompany'].num_nodes + \
#         orther['g_sector'].num_nodes + orther['h_area'].num_nodes
orther_count = []
for i in range(len(data)):
    count = data[i]['d_holdCompany'].num_nodes + data[i]['e_staff'].num_nodes + data[i]['f_supplyCompany'].num_nodes + \
            data[i]['g_sector'].num_nodes + data[i]['h_area'].num_nodes
    orther_count.append(count)
#
# metapath_for_rw = [[('a_manage', 'hold', 'd_holdCompany'), ('d_holdCompany', 'rev_hold', 'a_manage')],
#                    [('a_manage', 'exploy', 'e_staff'), ('e_staff', 'rev_exploy', 'a_manage')],
#                    [('b_produce', 'need', 'f_supplyCompany'), ('f_supplyCompany', 'rev_need', 'b_produce')],
#                    [('c_environment', 'belong', 'g_sector'), ('g_sector', 'rev_belong', 'c_environment')],
#                    [('c_environment', 'locate', 'h_area'), ('h_area', 'rev_locate', 'c_environment')]]
metapath_for_rw = [[('abc_stock', 'hold', 'd_holdCompany'), ('d_holdCompany', 'rev_hold', 'abc_stock')],
                   [('abc_stock', 'exploy', 'e_staff'), ('e_staff', 'rev_exploy', 'abc_stock')],
                   [('abc_stock', 'need', 'f_supplyCompany'), ('f_supplyCompany', 'rev_need', 'abc_stock')],
                   [('abc_stock', 'belong', 'g_sector'), ('g_sector', 'rev_belong', 'abc_stock')],
                   [('abc_stock', 'locate', 'h_area'), ('h_area', 'rev_locate', 'abc_stock')]]

random_modules = nn.ModuleList()
for i in range(len(data)):
    for j in range(len(metapath_for_rw)):
        random_module = notdivision_randomWalk(data[i], data[i].edge_index_dict, embedding_dim=62,
                                   metapath=metapath_for_rw[j], walk_length=5, context_size=5,
                                   walks_per_node=15, num_negative_samples=2,
                                   sparse=True)
        random_modules.append(random_module)
#
loaders = []
for num in random_modules:
    loader = num.loader(batch_size=128, shuffle=False, num_workers=0)
    loaders.append(loader)

#
# # step 2
# # 转换一下数据集以放进HAN训练
# # 用于HAN训练的METAPATH
# metapaths = [[('a_manage', 'd_holdCompany'), ('d_holdCompany', 'a_manage')],
#              [('a_manage', 'e_staff'), ('e_staff', 'a_manage')],
#              [('b_produce', 'f_supplyCompany'), ('f_supplyCompany', 'b_produce')],
#              [('c_environment', 'g_sector'), ('g_sector', 'c_environment')],
#              [('c_environment', 'h_area'), ('h_area', 'c_environment')]]
metapaths = [[('abc_stock', 'd_holdCompany'), ('d_holdCompany', 'abc_stock')],
             [('abc_stock', 'e_staff'), ('e_staff', 'abc_stock')],
             [('abc_stock', 'f_supplyCompany'), ('f_supplyCompany', 'abc_stock')],
             [('abc_stock', 'g_sector'), ('g_sector', 'abc_stock')],
             [('abc_stock', 'h_area'), ('h_area', 'abc_stock')]]
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                           drop_unconnected_node_types=True)

transformed_data = []  # 获取由元路径连接的数据
for i in range(len(data)):
    temperal = transform(data[i])
    transformed_data.append(temperal)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# step 3

model = HANLSTM(in_channels = -1 , out_channels=16,hidden_channels=16, input_size=16, hidden_size=16,
                num_layers=2,output_size=16, dataset=transformed_data, orther_count=orther_count)


model = model.to(device)
for x in data:
    x = x.to(device)
for x in transformed_data:
    x = x.to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train(loaders):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(loaders)
    loss.backward()
    optimizer.step()
    return float(loss)




# for epoch in range(1, 10):
#     loss = train(loaders)
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#
#
#
# torch.save(model.state_dict(), 'test.1_model_weights.pth')

model.load_state_dict(torch.load('no.3_model_weights.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for x in data:
    x = x.to(device)
model.eval()
with torch.no_grad():
    embedding_result,_ = model(data)

embedding_result = embedding_result[-1,:,:]
embedding_df = pd.DataFrame(embedding_result.cpu().numpy())
print("成功")
embedding_df.to_csv('embedding_result_3.csv',index=False)