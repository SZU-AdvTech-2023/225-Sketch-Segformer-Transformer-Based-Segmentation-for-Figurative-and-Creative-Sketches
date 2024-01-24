import os
import time
from options import TrainOptions, TestOptions
# from framework import SketchModel
from my_model import SketchModel
from utils import load_data
from writer import Writer
from evaluate import run_eval
import show
import json
from mydataset02 import my_datalodaer
import torch

import numpy as np
# import torchsnooper


def run_train(train_params=None, test_params=None):
    opt = TrainOptions().parse(train_params)
    testopt = TestOptions().parse(test_params)
    testopt.timestamp = opt.timestamp
    testopt.batch_size = opt.batch_size

    # model init
    model = SketchModel(opt)
    model.print_detail()

    writer = Writer(opt)

    # data load
    # trainDataloader = load_data(opt, datasetType='train', shuffle=opt.shuffle)
    # testDataloader = load_data(opt, datasetType='test')
    trainDataloader = my_datalodaer(opt, type='train', shuffle=opt.shuffle)
    testDataloader = my_datalodaer(opt, type='test')

    # train epoches
    # with torchsnooper.snoop():
    ii = 0
    min_test_avgloss = 100
    min_test_avgloss_epoch = 0
    for epoch in range(opt.epoch):
        for i, data in enumerate(trainDataloader):
            # torch.autograd.set_detect_anomaly(True)
            model.step(data)
            if ii % opt.plot_freq == 0:
                writer.plot_train_loss(model.loss, ii)
            if ii % opt.print_freq == 0:
                writer.print_train_loss(epoch, i, model.loss)

            ii += 1

        model.update_learning_rate()
        if opt.plot_weights:
            writer.plot_model_wts(model, epoch)
        
        # test
        if epoch % opt.run_test_freq == 0:
            model.save_network('latest')
            loss_avg, P_metric, C_metric = run_eval(
                opt=testopt,
                loader=testDataloader, 
                dataset='test',
                write_result=False)
            writer.print_test_loss(epoch, 0, loss_avg)
            writer.plot_test_loss(loss_avg, epoch)
            writer.print_eval_metric(epoch, P_metric, C_metric)
            writer.plot_eval_metric(epoch, P_metric, C_metric)
            if loss_avg < min_test_avgloss:
                min_test_avgloss = loss_avg
                min_test_avgloss_epoch = epoch
                print('saving the model at the end of epoch {} with test best avgLoss {}'.format(epoch, min_test_avgloss))
                model.save_network('bestloss')

    testopt.which_epoch = 'latest'
    testopt.metric_way = 'wlen'
    loss_avg, P_metric, C_metric = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)
    
    testopt.which_epoch = 'bestloss'
    testopt.metric_way = 'wlen'
    loss_avg_2, P_metric_2, C_metric_2 = run_eval(
        opt=testopt,
        loader=testDataloader, 
        dataset='test',
        write_result=False)

    record_list = {
        'p_metric': round(P_metric*100, 2),
        'c_metric': round(C_metric*100, 2),
        'loss_avg': round(loss_avg, 4),
        'best_epoch': min_test_avgloss_epoch,
        'p_metric_2': round(P_metric_2*100, 2),
        'c_metric_2': round(C_metric_2*100, 2),
        'loss_avg_2': round(loss_avg_2, 4),
    }
    writer.train_record(record_list=record_list)
    writer.close()
    return record_list, opt.timestamp,opt




if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    record_list, _,opt = run_train()
    print(record_list)

    # 可视化原始数据和结果数据
    # record.ndjson:{"timestamp": "Nov29_13_42_03", "class_name": "airplane", "net_name": "Sketch-Segformer"}
    print("_________________________________________________________________")
    print("开始进行图像可视化")
    opt = TrainOptions().parse(None)
    # 1.获取记录模型最佳训练效果的记录文件的地址
    record_path = "./{}/{}/{}/record.ndjson".format(opt.checkpoints_dir,opt.dataset,opt.class_name)
    # record_path = "./checkpoints/train/alarm_clock/record.ndjson"

    # 2.打开记录文件并且获取相应的属性值
    with open(record_path, 'rb') as file:
        # 读取和解析 JSON 数据
        # data = json.load(file)
        file.seek(0, os.SEEK_END)  # 移动到文件的末尾
        end_byte = file.tell()  # 获取当前文件指针的位置
        file.seek(max(end_byte - 1024, 0))  # 向前移动一定的字节数，这里假设最后一行不会超过1024字节
        last_lines = file.readlines()  # 读取最后部分的所有行
        last_lines = last_lines[-1].decode()  # 解码最后一行为字符串
        data= json.loads(last_lines)
        # 提取所需的属性值
        timestamp = data["timestamp"]
        class_name = data["class_name"]
        net_name = data["net_name"]
    # 3.构建pkl地址
    pkl_path = "./{}/{}/{}/{}/{}_bestloss.pkl".format(opt.checkpoints_dir,opt.dataset,class_name, timestamp, net_name)
    show_test = show.Show(opt, pkl_path)
    show_test.show()

