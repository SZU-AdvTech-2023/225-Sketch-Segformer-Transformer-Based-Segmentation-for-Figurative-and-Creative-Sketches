# Sketch-Segformer
Codes for the paper: "Sketch-Segformer: Transformer-Based Segmentation for Figurative and Creative Sketches" (IEEE TIP)

## 环境依赖

- Pytorch>=1.6.0
- pytorch_geometric>=1.6.1
- tensorboardX>=1.9

## 数据集

- sketch-gnn项目中的的的SPG256数据集
- https://github.com/sYeaLumin/SketchGNN/tree/main/data/SPG256


## 代码运行

- 使用mask的方式实现stroke-level的自注意力机制:mynet.py中Sketch_SegFormer类别中使用Dual_self_attn这个双分支类别即可
- 使用图的消息传递方式实现现现stroke-level的自注意力机制:上述双分支类替换成Dual_self_attn03即可
- 更改训练数据类别：通过option文件中的dataset、classname等属性值即可
- 训练结果可视化：运行show.py
- 所有类别训练数据可视化：运行show_loss.py文件