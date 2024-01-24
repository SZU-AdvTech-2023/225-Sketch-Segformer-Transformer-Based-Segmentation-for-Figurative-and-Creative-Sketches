import os
import copy
import json
import math
import random
import traceback

import torch
import numpy as np
from scipy.spatial import KDTree
from torch_geometric.data import Data
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString, Point
from rdp import rdp
from scipy.interpolate import interp1d
import os.path as osp
class SketchData(Data):
    def __init__(self, stroke_idx=None, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, norm=None, face=None, **kwargs):
        super(SketchData, self).__init__(x, edge_index, edge_attr, y=y, pos=pos, **kwargs)
        self.stroke_idx = stroke_idx
        self.stroke_num=1



def pad_edge_index(edge_index, max_length, pad_value=-1):
    # 获取原始 edge_index 的大小
    current_length = edge_index.size(1)

    # 计算需要填充的长度
    padding_length = max_length - current_length

    # 如果需要填充
    if padding_length > 0:
        # 创建填充用的张量
        padding = torch.full((2, padding_length), pad_value, dtype=edge_index.dtype)
        # 沿着列方向填充（堆叠）
        padded_edge_index = torch.cat([edge_index, padding], dim=1)
    else:
        padded_edge_index = edge_index

    return padded_edge_index


class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, root, class_name, split='train'):
        self.class_name = class_name
        self.split = split
        self.pt_dir = osp.join(root, '{}_{}.pt'.format(self.class_name, self.split)).replace("\\", "/")
        self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.class_name, self.split)).replace("\\", "/")
        # self.json_dir = osp.join(root, '{}_{}.ndjson'.format(self.class_name, self.split))
        self.out_segment = opt.out_segment

        # self.edge_index_max_len=-1
        # # print(self.edge_index_max_len)
        # self.pool_edge_index_max_len=-1
        # #
        if os.path.exists(self.pt_dir):
            self.processed_data = torch.load(self.pt_dir)
        else:
            self.processed_data = self.my_process()


    def __getitem__(self, index):
        return self.processed_data[index]

    def __len__(self):
        return len(self.processed_data)

    def _scale_sketch(self, sketchArray, target_size=(256, 256)):
        # 将所有笔画的坐标合并为一个数组
        all_strokes = np.concatenate(sketchArray, axis=1)  # 假设每个笔画是一个 3xN 的数组

        # 提取所有点的 x 和 y 坐标
        all_xy_coords = all_strokes[:2, :]

        # 找到 x 和 y 坐标的最大值和最小值
        min_coords = np.min(all_xy_coords, axis=1, keepdims=True)
        max_coords = np.max(all_xy_coords, axis=1, keepdims=True)

        # 计算缩放比例
        # scales = np.array(target_size).reshape(-1, 1) / (max_coords - min_coords)
        scales = (max_coords - min_coords) # 将会将坐标缩放到0到1


        # 缩放所有的点
        scaled_sketch = []
        for stroke in sketchArray:
            scaled_xy = (stroke[:2, :] - min_coords) * scales
            scaled_stroke = np.vstack([scaled_xy, stroke[2:, :]])  # 第三行是 label
            scaled_sketch.append(scaled_stroke)

        return scaled_sketch,all_xy_coords
        pass


    def _apply_rdp_and_resample(self, sketchArray, total_num_points=256, epsilon=2):
        # 先合并所有原始笔画
        all_original_strokes = np.concatenate([stroke[:2, :] for stroke in sketchArray], axis=1)
        all_original_labels_and_stroke_idx = np.concatenate([stroke[2:, :] for stroke in sketchArray], axis=1)

        # KD-Tree 基于原始点构建
        tree = KDTree(all_original_strokes.T)

        # 应用 RDP 算法简化
        simplified_sketch = [rdp(stroke[:2, :].T, epsilon=epsilon).T for stroke in sketchArray]
        all_strokes = np.concatenate(simplified_sketch, axis=1)

        resampled_strokes = self._interpolate_stroke(all_strokes, total_num_points, tree,all_original_labels_and_stroke_idx)

        # 重新分配点回笔画，并保留对应的标签
        resampled_sketch= []
        for i in np.unique(resampled_strokes[3]):
            stroke_mask=resampled_strokes[3]==i
            resampled_sketch.append(resampled_strokes[:,stroke_mask])

        return resampled_sketch

    def _interpolate_stroke(self, stroke, num_points, tree, all_labels_and_stroke_idx):
        x = np.linspace(0, 1, stroke.shape[1])
        x_new = np.linspace(0, 1, num_points)
        interp_func = interp1d(x, stroke[:2, :], kind='linear', axis=1)
        interpolated_points = interp_func(x_new)

        # 为插值点分配基于整个草图最近邻的标签
        interpolated_labels = []
        interpolated_stroke_idx=[]
        for point in interpolated_points.T:
            dist, index = tree.query(point)
            interpolated_labels.append(all_labels_and_stroke_idx[0,index])
            interpolated_stroke_idx.append(all_labels_and_stroke_idx[1,index])

        return np.vstack([interpolated_points, interpolated_labels,interpolated_stroke_idx])


    def my_process(self):
        raw_data = [] # 获取数据集中jison文件中的sketch数据
        with open(self.json_dir, 'r') as f:
            for line in f:
                raw_data.append(json.loads(line)["drawing"])
        processed_data = []

        for idx, sketch in enumerate(raw_data):
            sketchArray = [np.array(s) for s in sketch] #将数据转成array格式

            # 添加stroke_idx信
            # stroke_idx = np.concatenate([np.zeros(len(s[0])) + i for i, s in enumerate(sketchArray)])
            # sketchArray = np.concatenate([s for s in sketchArray], axis=1)  # 合并为 3xN 的数组
            # sketchArray = np.vstack([sketchArray, stroke_idx])  # 变为 4xN 的数组
            for i, s in enumerate(sketchArray):
                stroke_idx=np.zeros(len(s[0])) + i # 为每个点创建一个所属的笔画序号
                sketchArray[i]=np.vstack((sketchArray[i],stroke_idx))

            # 缩放sketch 256*256
            sketchArray,org_points = self._scale_sketch(sketchArray)

            # RDP简化sketch 并且 最后resample为256个点
            sketchArray= self._apply_rdp_and_resample(sketchArray)
            print("_apply_rdp_and_resample执行了第{}轮".format(idx))
            # 后续数值处理
            stroke_idx = np.concatenate([s[3] for s in sketchArray], axis=0) # 所在的stroke索引
            point = np.concatenate([s.transpose()[:,:2] for s in sketchArray]) # x和y坐标构成的点坐标，已经缩放到了256*256
            label = np.concatenate([s[2] for s in sketchArray], axis=0) # (N, ) # 人工标注的label

            # edge_index
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


            sketch_data = SketchData(x=torch.FloatTensor(point),
                                    org_x=torch.FloatTensor(point),
                                    edge_index=torch.LongTensor(edge_index),
                                    stroke_idx=torch.LongTensor(stroke_idx),
                                    y=torch.LongTensor(label),
                                    pool_edge_index=torch.LongTensor(pool_edge_index),
                                    )
            processed_data.append(sketch_data)
        torch.save(processed_data, self.pt_dir)
        return processed_data




