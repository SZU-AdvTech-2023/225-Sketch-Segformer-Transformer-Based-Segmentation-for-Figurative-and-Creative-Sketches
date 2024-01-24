import os.path as osp
import copy
import json
import math
import random
import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import dataset
import framework
# from utils import draw_sketch
import mynet
from options import TrainOptions, TestOptions
from utils import load_data
import os.path as osp
import os


class Show():
    def __init__(self,opt,pkl_path,):
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.class_name=opt.class_name
        self.ndj_path=osp.join("./data/dataset",opt.dataset,"{}_valid.ndjson".format(self.class_name)).replace("\\","/") # 验证集的位置
        self.validDataloader=load_data(opt, datasetType='valid')

        self.pkl_path=pkl_path
        self.net=mynet.init_net(opt)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
        print('loading the model from {}'.format(self.pkl_path))
        state_dict = torch.load(self.pkl_path,map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        self.net.load_state_dict(state_dict)
        self.net.eval()


    def draw_sketch(self,sketch,index,data_type="org_data"):
        save_path = "./res_my/{}/{}".format(self.class_name,data_type)
        # 检查路径是否存在，如果不存在，则创建它
        if not osp.exists(save_path):
            os.makedirs(save_path)
        save_file="{}/{}_{}.png".format(save_path,self.class_name,index)

        plt.figure(figsize=(10, 10))
        # plt.figure(figsize=(256/72, 256/72), dpi=72)
        color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple', 5: 'brown'}

        # 遍历每个笔画
        for stroke in sketch:
            x, y, labels = stroke
            if data_type == "org_image":
                # print(labels[0])
                plt.plot(x, y, color=color_map.get(labels[0],"black"),marker='o')  # 简化示例，不考虑标签
            else:
                # 预测结果的sketch的label被替换了，现在是tensor格式，需要使用数值，因此单独考虑这种情况
                # print(labels[0].data.item())
                plt.plot(x, y, color=color_map.get(labels[0].data.item(),"black"),marker='o')  # 简化示例，不考虑标签

        # 设置画布属性
        plt.xlim(0, 256)
        plt.ylim(0, 256)
        plt.gca().invert_yaxis()  # Y轴反转，以匹配常见的绘图坐标系统
        # 自动调整布局
        # # 关闭坐标轴
        plt.axis('off')
        plt.savefig(save_file,bbox_inches='tight')
        plt.close()



    def show(self):

        raw_data = []  # ndjson文件中的所有sketch
        with open(self.ndj_path, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)["drawing"])

        # for idx, (org_sketch, process_data) in enumerate(zip(raw_data, self.validDataloader)):
        for idx, org_sketch in enumerate(raw_data):
            # 绘制数据集中的图像
            self.draw_sketch(org_sketch,idx,data_type="org_image")

            # 替换label为模型推理的结果
            process_data=self._process(org_sketch)

            # 数据预处理
            stroke_data = {}
            x = process_data.x.to(self.device).requires_grad_(False)
            label = process_data.y.to(self.device)
            edge_index = process_data.edge_index.to(self.device)
            stroke_data['stroke_idx'] = process_data.stroke_idx.to(self.device)
            stroke_data['pool_edge_index'] = process_data.pool_edge_index.to(self.device)
            stroke_data['pos'] = x

            # 获得模型输出
            out = self.net(x.data, stroke_data)
            # 得到每个点相应的标签值
            _, predicted_labels = torch.max(out, 1)
            # 改变一下数值，测试
            # predicted_labels[0:22]=5
            # 替换原始数据中的label值，作为绘制结果的sketch
            res_sketches=org_sketch
            start=0
            for stroke in res_sketches:
                LEN=len(stroke[2])
                stroke[2]=predicted_labels[start:start+len(stroke[2])]
                start += len(stroke[2])
            # 绘制并且保存结果图
            self.draw_sketch(res_sketches,idx,data_type="res_data")
            pass
        print("{}类别的org_images和res_images保存完毕".format(self.class_name,))

    def _process(self,sketch):
        # processed_data=[]

        sketchArray = [np.array(s) for s in sketch]
        stroke_idx = np.concatenate([np.zeros(len(s[0])) + i for i, s in enumerate(sketchArray)])
        point = np.concatenate([s.transpose()[:, :2] for s in sketchArray])
        # normalize the data (N x 2)
        point = point.astype(np.float)
        max_point = np.max(point, axis=0)
        min_point = np.min(point, axis=0)
        org_point = point
        point = (point - min_point) / (max_point - min_point)
        label = np.concatenate([s[2] for s in sketchArray], axis=0)
        edge_index = []
        s = 0
        for stroke in sketchArray:
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
        for stroke in sketchArray:
            for i in range(len(stroke[0])):
                pool_edge_index.append([s + i, s + i])  # self loop
                for j in range(i + 1, len(stroke[0])):
                    pool_edge_index.append([s + i, s + j])
                    pool_edge_index.append([s + j, s + i])
            s += len(stroke[0])
        pool_edge_index = np.array(pool_edge_index).transpose()

        sketch_data = dataset.SketchData(x=torch.FloatTensor(point),
                                         org_x=torch.FloatTensor(org_point),
                                         edge_index=torch.LongTensor(edge_index),
                                         stroke_idx=torch.LongTensor(stroke_idx),
                                         y=torch.LongTensor(label),
                                         pool_edge_index=torch.LongTensor(pool_edge_index),
                                         )
        return sketch_data


# opt = TrainOptions().parse(None)
# show_test=Show(opt,"./checkpoints/train/airplane/Nov29_13_42_03/Sketch-Segformer_bestloss.pkl")
# show_test.show()
