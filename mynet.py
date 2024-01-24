import numpy as np
import torch.nn as nn
import torch
from torch_geometric.nn.conv import MessagePassing
import torch_geometric as tg


def init_net(opt):
    if opt.net_name == 'Sketch-Segformer':
        net = Sketch_Segformer(opt)
    else:
        raise NotImplementedError('net {} is not implemented. Please check.\n'.format(opt.net_name))

    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(opt.gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.gpu_ids)
    return net

class GA_Block(nn.Module):
    def __init__(self,channel,qk_channel,v_channel):
        super(GA_Block, self).__init__()
        self.scale=qk_channel**0.5 # qk通道数的平方根
        self.channel=channel
        self.q_metrix=nn.Conv1d(channel,qk_channel,1,bias=False)
        self.k_metrix=nn.Conv1d(channel,qk_channel,1,bias=False)
        self.k_metrix.weight.data=self.q_metrix.weight.data.clone() # q,k矩阵参数相同
        self.v_metrix=nn.Conv1d(channel,v_channel,1)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        # x=data.x
        x_q=self.q_metrix(x).permute(0,2,1) # b,n,c
        x_k=self.k_metrix(x) # b,n,c
        x_v=self.v_metrix(x).permute(0,2,1) # b,n,c

        energy=torch.bmm(x_q,x_k)/self.scale
        attention=self.softmax(energy)

        x=torch.bmm(attention,x_v).permute(0,2,1) # b,c,n
        return x

class SA_Block(nn.Module):
    def __init__(self,channel,qk_channel,v_channel):
        super(SA_Block, self).__init__()
        self.scale = qk_channel ** 0.5  # qk通道数的平方根
        self.channel = channel
        self.q_metrix = nn.Conv1d(channel, qk_channel, 1, bias=False)
        self.k_metrix = nn.Conv1d(channel, qk_channel, 1, bias=False)
        self.k_metrix.weight.data = self.q_metrix.weight.data.clone()  # q,k矩阵参数相同
        self.v_metrix = nn.Conv1d(channel, v_channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,stroke_data):
        final_x=torch.zeros((64,32,256)).to(x.device)
        org_x = x
        for i in range(stroke_data["stroke_idx"][-1]):
            mask=stroke_data["stroke_idx"]==i  # (64*256,)
            x=org_x
            x=x.permute(0,2,1).contiguous().view(-1,64)*mask.contiguous().view(-1,1)# (64,64,256)-->(64,256,64)-->(64*256,64)
            # np_mask=mask.cpu().numpy()
            # np_x=x.cpu().detach().numpy()
            stroke_x=x.contiguous().view(64,256,64).permute(0,2,1) # (64*256,64)-->(64,256,64)-->(64,64,256)
            x_q = self.q_metrix(stroke_x).permute(0, 2, 1)
            x_k = self.k_metrix(stroke_x)
            x_v = self.v_metrix(stroke_x).permute(0, 2, 1)

            energy = torch.bmm(x_q, x_k) / self.scale
            attention = self.softmax(energy)

            x = torch.bmm(attention,x_v).permute(0,2,1) # b,c,n
            final_x = final_x+x  # 需要使用新的地址存放，否则梯度计算时会出现错误
            # np_x = x.cpu().detach().numpy()
        # concat_x=np.concatenate(final_x,axis=2)
        return final_x

class SA_Block_02(nn.Module):
    def __init__(self, channel, qk_channel, v_channel):
        super(SA_Block_02, self).__init__()
        self.scale = qk_channel ** 0.5  # qk通道数的平方根
        self.channel = channel
        self.q_metrix = nn.Conv1d(channel, qk_channel, 1, bias=False)
        self.k_metrix = nn.Conv1d(channel, qk_channel, 1, bias=False)
        self.k_metrix.weight.data = self.q_metrix.weight.data.clone()  # q,k矩阵参数相同
        self.v_metrix = nn.Conv1d(channel, v_channel, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, stroke_data):
        "用类似masked attention的方式实现，创建的mask类似与padding mask，不相干的值的位置的mask值为负无穷"
        # final_x = torch.zeros((64, 32, 256)).to(x.device)
        final_x=[]
        org_x = x
        # stroke_idx的值是从0开始
        for i in range(torch.max(stroke_data["stroke_idx"])+1):
            stroke_idx = stroke_data["stroke_idx"]
            # 创建一个与 stroke_idx 形状相同的 mask
            mask = torch.zeros_like(stroke_idx, dtype=torch.float)
            # 设置满足条件的位置为 0，不满足条件的位置为 -inf
            mask[stroke_idx == i] = 0
            # mask[stroke_idx != i] = float('-inf') # (16384,)
            "直接赋值为-inf的话，会导致softmax操作之后得到nan的值"
            mask[stroke_idx != i] = 10e-7 # (16384,)

            np_mask=mask.contiguous().view(-1,256).cpu().numpy()
            mask=mask.contiguous().view(-1,256).unsqueeze(1).repeat(1,256,1) # (16384,)-->(b,256,256)

            x = org_x
            x = x.permute(0, 2, 1).contiguous().view(-1, 64)  # (b,64,256)-->(b,256,64)-->(b*256,64)
            # np_x=x.cpu().detach().numpy()
            stroke_x = x.contiguous().view(-1, 256, 64).permute(0, 2, 1)  # (b*256,64)-->(b,256,64)-->(b,64,256)
            x_q = self.q_metrix(stroke_x).permute(0, 2, 1)
            x_k = self.k_metrix(stroke_x)
            x_v = self.v_metrix(stroke_x).permute(0, 2, 1)

            energy = ((torch.bmm(x_q, x_k) / self.scale)+mask)*10# (b,256,256)
            "有可能导致energy中的值过小，softmax之后产生了nan值"
            # if torch.isnan(energy).any() :
            #     print("Energy tensor contains NaNs or all -inf")

            attention = self.softmax(energy)
            # if torch.isnan(attention).any():
            #     print("NaN values found in attention tensor after softmax")

            x = torch.bmm(attention, x_v).permute(0, 2, 1)  # b,c,n
            # print("x的{}".format(torch.isnan(x).all()))

            final_x.append(x)  # 需要使用新的地址存放，否则梯度计算时会出现错误
            # np_x = x.cpu().detach().numpy()
        stacked_x = torch.stack(final_x, dim=0)
        summed_x = torch.sum(stacked_x, dim=0)
        # print(torch.isnan(summed_x).all())
        return summed_x


class SA_Block_03(MessagePassing):
    def __init__(self, channels, qk_channels, v_channels, aggr='add', **kwargs):
        super(SA_Block_03, self).__init__(aggr=aggr, **kwargs)
        # self.nn = nn
        self.temperature = qk_channels ** 0.5
        self.channels = channels
        self.q_matrix = nn.Linear(channels, qk_channels, bias=False)
        self.k_matrix = nn.Linear(channels, qk_channels, bias=False)
        self.q_matrix.weight.data = self.k_matrix.weight.data.clone()
        # self.q_matrix.weight = self.k_matrix.weight
        # self.q_matrix.bias = self.k_matrix.bias
        self.v_matrix = nn.Linear(channels, v_channels, bias=False)
        self.aggr = aggr
        self.edge_index = None

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x  # (64*256,64)
        self.edge_index = edge_index
        out = self.propagate(edge_index, x=x)  # (64*256,32)
        self.edge_index = None
        # x = x + out

        return out

    def message(self, x_i, x_j):
        x_q = self.q_matrix(x_i)
        x_k = self.k_matrix(x_j)
        x_v = self.v_matrix(x_j)
        energy = x_q * x_k
        energy = torch.sum(energy, dim=1, keepdim=True)
        energy = energy / self.temperature
        attention = tg.utils.softmax(energy, self.edge_index[1])

        self.edge_index = None

        return x_v * attention


class Dual_Slef_Attn(nn.Module):
    def __init__(self,channel):
        super(Dual_Slef_Attn, self).__init__()
        self.channel=channel

        self.ga=GA_Block(self.channel, self.channel//2//4, self.channel//2)
        self.ga_lb=nn.Sequential(nn.Conv1d(self.channel,self.channel//2,kernel_size=1,bias=False),
                                 nn.BatchNorm1d(self.channel//2))
        self.ga_bn=nn.BatchNorm1d(self.channel//2)

        self.sa = SA_Block_02(self.channel, self.channel // 2 // 4, self.channel // 2)
        self.sa_lb = nn.Sequential(nn.Conv1d(self.channel, self.channel // 2,kernel_size=1,bias=False),
                                   nn.BatchNorm1d(self.channel // 2))
        self.sa_bn = nn.BatchNorm1d(self.channel // 2)

        self.mlp=nn.Sequential(nn.Conv1d(self.channel, self.channel*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channel*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channel*4, self.channel, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channel),
                                )

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x,stroke_data):
        f_ga=self.ga_bn(self.ga(x))+self.ga_lb(x)
        f_sa=self.sa_bn(self.sa(x,stroke_data))+self.sa_lb(x)
        f= torch.cat((f_ga, f_sa), dim=1) # b,c,n
        f_mlp=self.mlp(f)
        f_out=self.relu(f_mlp+f)
        torch.cuda.empty_cache()
        return f_out

class Dual_Slef_Attn_03(nn.Module):
    def __init__(self,channel):
        super(Dual_Slef_Attn_03, self).__init__()
        self.channel=channel

        self.ga=GA_Block(self.channel, self.channel//2//4, self.channel//2)
        self.ga_lb=nn.Sequential(nn.Conv1d(self.channel,self.channel//2,kernel_size=1,bias=False),
                                 nn.BatchNorm1d(self.channel//2))
        self.ga_bn=nn.BatchNorm1d(self.channel//2)

        self.sa = SA_Block_03(self.channel, self.channel // 2 // 4, self.channel // 2)
        self.sa_lb = nn.Sequential(nn.Conv1d(self.channel, self.channel // 2,kernel_size=1,bias=False),
                                   nn.BatchNorm1d(self.channel // 2))
        self.sa_bn = nn.BatchNorm1d(self.channel // 2)

        self.mlp=nn.Sequential(nn.Conv1d(self.channel, self.channel*4, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channel*4),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(self.channel*4, self.channel, kernel_size=1, bias=False),
                                nn.BatchNorm1d(self.channel),
                                )

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x,stroke_data):
        batch_size, _, N = x.size()
        f_ga=self.ga_bn(self.ga(x))+self.ga_lb(x)
        f_sa=self.sa_bn(self.sa(x.permute(0,2,1).contiguous().view(-1, self.channel),torch.unique(stroke_data["pool_edge_index"],dim=1)).view(batch_size, 256, -1).permute(0, 2, 1))+self.sa_lb(x)
        f= torch.cat((f_ga, f_sa), dim=1) # b,c,n
        f_mlp=self.mlp(f)
        f_out=self.relu(f_mlp+f)
        torch.cuda.empty_cache()
        return f_out


class Sketch_Segformer(nn.Module):
    def __init__(self,opt):
        super(Sketch_Segformer, self).__init__()
        self.points_num = opt.points_num
        self.channels = opt.channels * 2
        self.in_feature = opt.in_feature
        self.out_segment = opt.out_segment

        self.pos_encoding = nn.Embedding(self.points_num, self.channels)
        self.mlp1=nn.Sequential(nn.Conv1d(self.in_feature,self.channels,kernel_size=1,bias=False),
                               nn.BatchNorm1d(self.channels),
                               nn.ReLU(inplace=True),
                               nn.Conv1d(self.channels, self.channels,kernel_size=1,bias=False),
                               nn.BatchNorm1d(self.channels),
                               nn.ReLU(inplace=True)
                               )

        self.dual_block_1=Dual_Slef_Attn(self.channels)
        self.dual_block_2=Dual_Slef_Attn(self.channels)
        self.dual_block_3=Dual_Slef_Attn(self.channels)
        self.dual_block_4=Dual_Slef_Attn(self.channels)

        # self.dual_block_1 = Dual_Slef_Attn_03(self.channels)
        # self.dual_block_2 = Dual_Slef_Attn_03(self.channels)
        # self.dual_block_3 = Dual_Slef_Attn_03(self.channels)
        # self.dual_block_4 = Dual_Slef_Attn_03(self.channels)

        self.lbr1=nn.Sequential(nn.Conv1d(self.channels*4,self.channels*4*4,kernel_size=1,bias=False),
                                nn.BatchNorm1d(self.channels*4*4),
                                nn.ReLU(inplace=True))

        self.lbr2=nn.Sequential(nn.Conv1d(self.channels*4*4*3,self.channels*4,kernel_size=1,bias=False),
                                nn.BatchNorm1d(self.channels*4),
                                nn.ReLU(inplace=True))

        self.mlp2=nn.Sequential(nn.Conv1d(self.channels*4,self.channels*4//2,kernel_size=1,bias=False),
                               nn.BatchNorm1d(self.channels*4//2),
                               nn.ReLU(inplace=True),
                               nn.Conv1d(self.channels*4//2, self.out_segment,kernel_size=1,bias=False),)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,x,stroke_data):
        if len(x.shape) == 2:
            x = x.view(-1, self.points_num, self.in_feature) # (64,256,2)
        else:
            print("Attention!!!")
            import pdb;
            pdb.set_trace()
        x = x.permute(0, 2, 1) #相当于是(B,C,N) (64,2,256)
        batch_size, _, N = x.size()
        pos = torch.arange(0, self.points_num).repeat(batch_size).view(batch_size, self.points_num).to(device=x.device)
        pos = self.pos_encoding(pos) # 位置编码 O
        pos = pos.permute(0, 2, 1) # (64,64,256)
        x=self.mlp1(x)+pos # mlp(P)+O # (64,64,256)

        x1=self.dual_block_1(x,stroke_data)
        x2=self.dual_block_2(x1,stroke_data)
        x3=self.dual_block_2(x2,stroke_data)
        x4=self.dual_block_2(x3,stroke_data)

        x=torch.cat((x1,x2,x3,x4),dim=1)

        x=self.lbr1(x)

        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, self.points_num)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, self.points_num)
        x=torch.cat((x_max_feature,x,x_avg_feature),1)

        x=self.lbr2(x)

        x=self.mlp2(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, self.out_segment)
        x = self.LogSoftmax(x)

        return x







