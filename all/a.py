import scipy.io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

def Get_KFold(input, label, n_splits):
    num_samples = input.size(0)
    indices = torch.randperm(num_samples)  # 随机排列样本索引
    fold_size = num_samples // n_splits
    
    folds = []
    
    for i in range(n_splits):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = torch.cat([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        
        train_input = input[train_indices]
        test_input = input[test_indices]
        train_label = label[train_indices]
        test_label = label[test_indices]
        
        folds.append((train_input, test_input, train_label, test_label))
        
    return folds

def plot(data):

    fig, axs = plt.subplots(4, 1, figsize=(15, 10))

    for i in range(4):
        axs[i].plot(data[i].detach().numpy())
        axs[i].set_title(f'Subplot {i+1}')
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

def batch_norm_1d(data):
    # data: [..., 2100, 19]
    # 调整数据形状为 (..., 19, 2100) 以适应 BatchNorm1d
    data = data.permute(0, 2, 1)
    batch_norm = nn.BatchNorm1d(19)
    normalized_data = batch_norm(data).permute(0, 2, 1)
    
    return normalized_data
    
    
# 加载 .mat 文件
rawTrace_Person1 = scipy.io.loadmat('Person1/rawTracePerson1.mat')
rawTrace_Person2 = scipy.io.loadmat('Person2/rawTracePerson2.mat')
rawTrace_Person3 = scipy.io.loadmat('Person3/rawTracePerson3.mat')
rawTrace_Person4 = scipy.io.loadmat('Person4/rawTracePerson4.mat')

OS_Person1 = scipy.io.loadmat('Person1/OSPerson1.mat')
OS_Person2 = scipy.io.loadmat('Person2/OSPerson2.mat')
OS_Person3 = scipy.io.loadmat('Person3/OSPerson3.mat')
OS_Person4 = scipy.io.loadmat('Person4/OSPerson4.mat')

# [40, 2100, 19]
rawTrace_person1 = torch.Tensor(rawTrace_Person1['dataTrial']).permute(1, 0, 2)
# [10, 2100, 19]
rawTrace_person2 = torch.Tensor(rawTrace_Person2['dataTrial']).permute(1, 0, 2)
# [40, 2100, 19]
rawTrace_person3 = torch.Tensor(rawTrace_Person3['dataTrial']).permute(1, 0, 2)
# [10, 2100, 19]
rawTrace_person4 = torch.Tensor(rawTrace_Person4['dataTrial']).permute(1, 0, 2)

# [40, 36, 52, 54]
OS_person1 = torch.Tensor(OS_Person1['OS']).permute(2, 0, 1, 3)
# [10, 36, 52, 54]
OS_person2 = torch.Tensor(OS_Person2['OS']).permute(2, 0, 1, 3)
# [40, 36, 52, 54]
OS_person3 = torch.Tensor(OS_Person3['OS']).permute(2, 0, 1, 3)
# [10, 36, 52, 54]
OS_person4 = torch.Tensor(OS_Person4['OS']).permute(2, 0, 1, 3)

# [40,]
label_person1 = torch.Tensor(rawTrace_Person1['Track']).squeeze()
# [10,]
label_person2 = torch.Tensor(rawTrace_Person2['Track']).squeeze()
# [40,]
label_person3 = torch.Tensor(rawTrace_Person3['Track']).squeeze()
# [10,]
label_person4 = torch.Tensor(rawTrace_Person4['Track']).squeeze()

# [5, 4, ...] : 每个人的数据均分为 5 份，每次选一份作为测试集
# Useage :  train_input, test_input, train_label, test_label = rawTrace_folds1[i]
rawTrace_folds1 = Get_KFold(rawTrace_person1, label_person1, 5)
rawTrace_folds2 = Get_KFold(rawTrace_person2, label_person2, 5)
rawTrace_folds3 = Get_KFold(rawTrace_person3, label_person3, 5)
rawTrace_folds4 = Get_KFold(rawTrace_person4, label_person4, 5)

OS_folds1 = Get_KFold(OS_person1, label_person1, 5)
OS_folds2 = Get_KFold(OS_person2, label_person2, 5)
OS_folds3 = Get_KFold(OS_person3, label_person3, 5)
OS_folds4 = Get_KFold(OS_person4, label_person4, 5)


# ==================== test batch norm =========================

# # 调整数据形状为 (40, 19, 2100) 以适应 BatchNorm1d
# data_1 = rawTrace_person1.permute(0, 2, 1)
# data_2 = rawTrace_person2.permute(0, 2, 1)
# data_3 = rawTrace_person3.permute(0, 2, 1)
# data_4 = rawTrace_person4.permute(0, 2, 1)

# batch_norm = nn.BatchNorm1d(19)

# normalized_data1 = batch_norm(data_1).permute(0, 2, 1)
# normalized_data2 = batch_norm(data_2).permute(0, 2, 1)
# normalized_data3 = batch_norm(data_3).permute(0, 2, 1)
# normalized_data4 = batch_norm(data_4).permute(0, 2, 1)

# tmp = torch.stack((normalized_data1[2, :, 0], normalized_data2[2, :, 0],
#                    normalized_data3[2, :, 0], normalized_data4[2, :, 0]), dim = 0)
# print(plot(tmp))

# ==================== test batch norm =========================
