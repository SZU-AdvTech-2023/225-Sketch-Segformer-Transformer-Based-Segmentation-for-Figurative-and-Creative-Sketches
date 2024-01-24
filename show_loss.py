import os
import os.path as osp
import json

import matplotlib.pyplot as plt

from options import TrainOptions


def parse_loss(log_file):
    with open(log_file, 'r') as file:
        train_losses = []
        test_losses = []
        for line in file:
            # 假设每行包含“loss”和数值
            if 'loss' in line:
                parts = line.split()
                loss_index = parts.index('loss:') + 1
                loss_value = float(parts[loss_index])
                if line.startswith('test'):
                    test_losses.append(loss_value)
                else:
                    train_losses.append(loss_value)
        return train_losses, test_losses

def plot_and_save_losses(train_losses, test_losses,class_name):
    save_path = "./loss_my"
    file_name="{}/{}_loss.png".format(save_path,class_name)
    # test_file_name="{}/{}_test_loss.png".format(save_path,class_name)
    # 检查路径是否存在，如果不存在，则创建它
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # 绘制并保存训练损失图像
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Time')
    plt.legend()
    plt.savefig(file_name)  # 保存测试损失图像
    plt.close()  # 关闭当前图像


def show_all_best_loss(loss_data,type):
    # 找出每个类别的最小loss
    final_losses = {cls: losses[-1] if isinstance(losses, list) else losses for cls, losses in loss_data.items()}

    # 准备绘图数据
    categories = list(final_losses.keys())
    loss_values = list(final_losses.values())

    # 保存文件名
    save_path = "./loss_my"
    file_name="{}/all_{}.png".format(save_path,type)

    # 绘制条形图
    plt.figure(figsize=(10,6))
    bars = plt.bar(categories, loss_values, color='skyblue',width=0.5)
    plt.xticks(rotation=45) # 设置x轴标签角度
    # 在每个柱子上添加数值
    for bar, loss in zip(bars, loss_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{loss:.3f}',  # 格式化为三位小数
                 va='bottom', ha='center', fontsize=8, rotation=45)
    plt.xlabel('Category')
    plt.ylabel('{}'.format(type))
    plt.title('{} per Category_org'.format(type))
    # 调整子图布局
    plt.subplots_adjust(bottom=0.2)  # 增加底部留白
    # 调整y轴的上限以留出空间显示数值标签
    plt.ylim(0, max(loss_values) * 1.2)  # 将y轴上限设为最大值的120%
    plt.savefig(file_name)  # 保存测试损失图像
    plt.close()  # 关闭当前图像


def show_and_save_loss(class_list):
    total_train_loss={}
    total_test_loss={}
    total_p_metric={}
    total_c_metric={}
    opt = TrainOptions().parse(None)
    for class0 in class_list:
        # 1.获取记录模型最佳训练效果的记录文件的地址
        record_path = "./{}/train/{}/record.ndjson".format(opt.checkpoints_dir, class0)
        # record_path="./checkpoints_wosa/train/{}/record.ndjson".format(class0)

        # 2.打开记录文件并且获取相应的属性值
        with open(record_path, 'rb') as file:
            # 读取和解析 JSON 数据
            # data = json.load(file)
            file.seek(0, os.SEEK_END)  # 移动到文件的末尾
            end_byte = file.tell()  # 获取当前文件指针的位置
            file.seek(max(end_byte - 1024, 0))  # 向前移动一定的字节数，这里假设最后一行不会超过1024字节
            last_lines = file.readlines()  # 读取最后部分的所有行
            last_lines = last_lines[-1].decode()  # 解码最后一行为字符串
            data = json.loads(last_lines)
            # 提取所需的属性值
            timestamp = data["timestamp"]
            # p_metric=data["p_metric_2"]
            # c_metric=data["c_metric_2"]
            total_p_metric[class0]=data["p_metric_2"]
            total_c_metric[class0]=data["c_metric_2"]
        # 获得loss记录文件
        path = "./{}/train/{}/{}/loss_log.txt".format(opt.checkpoints_dir,class0,timestamp)
        # 分别读取出train 和 test情况下的loss
        train_loss,test_loss=parse_loss(path)
        # 显示并且保存相应的可视化图像
        plot_and_save_losses(train_loss,test_loss,class0)
        # 在所有loss的统计字典中加入属性和值
        total_train_loss[class0]=train_loss
        total_test_loss[class0]=test_loss

    # 绘制所有类别训练时的最小loss
    show_all_best_loss(total_train_loss,"min_train_loss")
    #
    show_all_best_loss(total_p_metric,"p_metric")
    show_all_best_loss(total_c_metric,"c_metric")


    return total_test_loss,total_train_loss





# log_file1 = 'checkpoints/train/airplane/Nov29_13_42_03/loss_log.txt'  # 替换为您的文件路径
# train_losses1, test_losses1 = parse_loss(log_file1)
# plot_losses(train_losses1, test_losses1)
#
# log_file2 = 'checkpoints_wosa/train/airplane/Dec05_05_53_45/loss_log.txt'  # 替换为您的文件路径
# train_losses2, test_losses2 = parse_loss(log_file2)
# plot_losses(train_losses2, test_losses2)
class_list=["airplane","alarm_clock","ambulance","ant","apple","backpack","basket","butterfly","cactus","calculator","campfire","candle","crab","coffee_cup","duck","face","ice_cream","pig","pineapple","suitcase"]
# class_list=["airplane","alarm_clock"]

show_and_save_loss(class_list)