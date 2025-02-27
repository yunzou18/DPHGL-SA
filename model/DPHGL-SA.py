import torch
import torch.nn as nn
from Model.HAN import HAN
from Model.LSTM import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
EPS = 1e-15
class HANLSTM(nn.Module):
    def __init__(self, in_channels,  out_channels,hidden_channels, input_size, hidden_size, num_layers, output_size,dataset,orther_count): #添加了dataset
        super(HANLSTM, self).__init__()
        self.dataset = dataset
        self.output_size = output_size

        # Create a list of HAN modules for each graph
        self.han_modules = nn.ModuleList([HAN(in_channels, out_channels,hidden_channels, dataset[i],orther_count[i]) for i in range(len(dataset))])

        # LSTM layer
        self.lstm = LSTM(input_size, hidden_size, num_layers, output_size)

    def forward(self, graphs):
        # Process each graph with HAN and store the outputs

        han_outputs_total = []
        others_embedding = []


        for graph, han_module in zip(graphs, self.han_modules):
            han_output,_ = han_module(graph)
            others_embedding.append(_)
            han_outputs_total.append(han_output)


        # Combine the outputs into a single tensor
        han_outputs = torch.stack(han_outputs_total, dim=0)

        # Pass the concatenated output through the LSTM
        lstm_output= self.lstm(han_outputs)

        return lstm_output,others_embedding

    def loss(self,loaders):
        totalembeddinglist= []
        total_loss = 0
        hanlstm_output , others_embedding = self.forward(self.dataset)

        # sum_abc_list = []
        # for i in range(len(self.dataset)):
        #     a_manage = hanlstm_output[i,0:3550,:]
        #     b_produce = hanlstm_output[i,3550:7100,:]
        #     c_environment = hanlstm_output[i,7100:10650,:]
        #     sum_abc = a_manage+b_produce+c_environment
        #     sum_abc_list.append(sum_abc)


        # print("运行到loss了吗")
        for i in range(len(self.dataset)):
            totalembedding = torch.cat((hanlstm_output[i,:,:] ,others_embedding[i]),dim=0)
            totalembeddinglist.append(totalembedding)

        for i in range(len(loaders)):
            for j,(pos_rw,neg_rw) in enumerate(loaders[i]):
                start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
                # Positive loss.
                #totalembeddinglist[int(i/5)](start)
                #totalembeddinglist[int(i/5)](rest.view(-1))
                the_shape = totalembeddinglist[int(i / 5)].shape[0]
                result_start_pos = totalembeddinglist[int(i/5)][start%the_shape]
                result_rest_pos = totalembeddinglist[int(i/5)][rest%the_shape]
                h_start = result_start_pos.view(pos_rw.size(0), 1,self.output_size)
                h_rest = result_rest_pos.view(pos_rw.size(0), -1,self.output_size)
                out = (h_start * h_rest).sum(dim=-1).view(-1)

                pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

                # Negative loss.
                start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
                result_start_neg = totalembeddinglist[int(i / 5)][start % the_shape]
                result_rest_neg = totalembeddinglist[int(i / 5)][rest % the_shape]
                h_start = result_start_neg.view(neg_rw.size(0), 1,
                                                     self.output_size)
                h_rest = result_rest_neg.view(neg_rw.size(0), -1,self.output_size)

                out = (h_start * h_rest).sum(dim=-1).view(-1)
                neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

                total_loss+=pos_loss
                total_loss+=neg_loss
        return total_loss

