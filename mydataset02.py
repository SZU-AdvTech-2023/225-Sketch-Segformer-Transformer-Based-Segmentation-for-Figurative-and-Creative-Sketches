from torch_geometric.data import Data
import torch
import os
import json
import numpy as np
from torch_geometric.data import DataLoader


class SketchData(Data):
    def __init__(self, stroke_idx, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        super(SketchData, self).__init__(x, edge_index, edge_attr, y=y, pos=pos, **kwargs)
        self.stroke_idx = stroke_idx
        self.stroke_num= 1

class SketchDataSet(torch.utils.data.Dataset):
    def __init__(self,opt,root,class_name,split="train",):
        self.class_name=class_name
        """读取数据集中的文件，获取数据"""
        #1. 构建数据集访问地址
        self.data_path="{}/{}/{}_{}.ndjson".format(root,opt.dataset,self.class_name,split)
        self.pkl_path="{}/{}/{}_{}.pt".format(root,opt.dataset,self.class_name,split) # 与处理好的数据文件
        #2. 获取数据
        if os.path.exists(self.pkl_path):
            self.processed_data=torch.load(self.pkl_path)
        else:
            self.processed_data=self.process()
        pass


    def __getitem__(self, item):
        return self.processed_data[item]
    pass

    def __len__(self):
        return len(self.processed_data)

    def process(self):
        # 1. 获取数据--打开访问地址的文件--获取drawing属性的值
        raw_sketch=[]
        with open(self.data_path, 'r') as f:
            for line in f:
                raw_sketch.append(json.loads(line)["drawing"]) #要先获取到了str再传给json.load()进行加载
        # 2. 对数据进行格式转换 -- raw_sketch中一个元素就是一张sketch
        processed_data=[]
        for sketch in raw_sketch:
            sketch_array=[np.array(s) for s in sketch] # 将所有笔画转换成array格式
            points = np.concatenate([np.column_stack((s[0, :], s[1, :])) for s in sketch_array]) # 拼接所有x y坐标
            points=points.astype(float) # 转换成浮点数类型
            min_xy=np.min(points,axis=0) # 沿着列方向找最小值
            max_xy=np.max(points,axis=0)
            org_point=points
            points=(points - min_xy) /(max_xy- min_xy) # 值缩放到0-1
            label=np.concatenate([s[2] for s in sketch_array],axis=0) # 每个点的标签值 (N,)
            stroke_idx=np.concatenate([np.zeros(len(s[0]))+i for i,s in enumerate(sketch_array)]) # 创建每个点所属笔画id

            # edge_index
            edge_index = []
            s = 0
            for stroke in sketch_array:
                # edge_index.append([s,s])
                for i in range(len(stroke[0]) - 1):
                    edge_index.append([s + i, s + i + 1])
                    edge_index.append([s + i + 1, s + i])
                # edge_index.append([s,s+len(stroke[0])-1])
                s += len(stroke[0])
            edge_index = np.array(edge_index).transpose()

            # pool_edge_index
            pool_edge_index = []
            s = 0
            for stroke in sketch_array:
                for i in range(len(stroke[0])):
                    pool_edge_index.append([s + i, s + i])  # self loop
                    for j in range(i + 1, len(stroke[0])):
                        pool_edge_index.append([s + i, s + j])
                        pool_edge_index.append([s + j, s + i])
                s += len(stroke[0])
            pool_edge_index = np.array(pool_edge_index).transpose() # (N,2) 构建的时有向全连接图

            sketch_data=SketchData(x=torch.FloatTensor(points),
                                    org_x=torch.FloatTensor(org_point),
                                    edge_index=torch.LongTensor(edge_index),
                                    stroke_idx=torch.LongTensor(stroke_idx),
                                    y=torch.LongTensor(label),
                                    pool_edge_index=torch.LongTensor(pool_edge_index),)
            processed_data.append(sketch_data)
        torch.save(processed_data,self.pkl_path)
        return processed_data
        pass


def my_datalodaer(opt,type="train",shuffle=False):
    data_set = SketchDataSet(
        opt=opt,
        root='./data/dataset',
        class_name=opt.class_name,
        split=type,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers
    )
    return data_loader



