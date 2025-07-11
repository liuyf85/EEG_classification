import scipy.io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA

path = './'
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
    # data: [2100,]
    plt.figure(figsize=(15, 5))
    plt.plot(data.numpy())
    plt.title('Array Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

def batch_norm_1d(data):
    # data: [..., 2100, 19]
    # 调整数据形状为 (..., 19, 2100) 以适应 BatchNorm1d
    data = data.permute(0, 2, 1)
    batch_norm = nn.BatchNorm1d(19)
    normalized_data = batch_norm(data).permute(0, 2, 1)
    
    return normalized_data

def batch_norm_2d(data):
    # Create a BatchNorm2d layer
    batch_norm = nn.BatchNorm2d(data.size(-1))  # data.size(1) gives the number of channels
    
    data = data.permute(0, 3, 1, 2)
    # Normalize the data
    normalized_data = batch_norm(data).permute(0, 2, 3, 1)
    
    return normalized_data.detach()

# 加载 .mat 文件
rawTrace_Person1 = scipy.io.loadmat(path + 'Person1/rawTracePerson1.mat')
rawTrace_Person2 = scipy.io.loadmat(path + 'Person2/rawTracePerson2.mat')
rawTrace_Person3 = scipy.io.loadmat(path + 'Person3/rawTracePerson3.mat')
rawTrace_Person4 = scipy.io.loadmat(path + 'Person4/rawTracePerson4.mat')

OS_Person1 = scipy.io.loadmat(path + 'Person1/OSPerson1.mat')
OS_Person2 = scipy.io.loadmat(path + 'Person2/OSPerson2.mat')
OS_Person3 = scipy.io.loadmat(path + 'Person3/OSPerson3.mat')
OS_Person4 = scipy.io.loadmat(path + 'Person4/OSPerson4.mat')


# [40, 2100, 19]
rawTrace_person1 = torch.Tensor(rawTrace_Person1['dataTrial']).permute(1, 0, 2)
# [10, 2100, 19]
rawTrace_person2 = torch.Tensor(rawTrace_Person2['dataTrial']).permute(1, 0, 2)
# [40, 2100, 19]
rawTrace_person3 = torch.Tensor(rawTrace_Person3['dataTrial']).permute(1, 0, 2)
# [10, 2100, 19]
rawTrace_person4 = torch.Tensor(rawTrace_Person4['dataTrial']).permute(1, 0, 2)
# [100, 2100, 19]
rawTrace_person_all = torch.cat((rawTrace_person1, rawTrace_person2, rawTrace_person3, rawTrace_person4), dim = 0)


# [40, 36, 52, 54]
OS_person1 = torch.Tensor(OS_Person1['OS']).permute(2, 0, 1, 3)
# [10, 36, 52, 54]
OS_person2 = torch.Tensor(OS_Person2['OS']).permute(2, 0, 1, 3)
# [40, 36, 52, 54]
OS_person3 = torch.Tensor(OS_Person3['OS']).permute(2, 0, 1, 3)
# [10, 36, 52, 54]
OS_person4 = torch.Tensor(OS_Person4['OS']).permute(2, 0, 1, 3)
# [100, 36, 52, 54]
OS_person_all = torch.cat((OS_person1, OS_person2, OS_person3, OS_person4), dim = 0) 

# [40,]
label_person1 = torch.Tensor(rawTrace_Person1['Track']).squeeze()
# [10,]
label_person2 = torch.Tensor(rawTrace_Person2['Track']).squeeze()
# [40,]
label_person3 = torch.Tensor(rawTrace_Person3['Track']).squeeze()
# [10,]
label_person4 = torch.Tensor(rawTrace_Person4['Track']).squeeze()


# 将Track数据转换为二值标签（记忆过的图片为1，未记忆过的图片为0）
label_person1 = torch.Tensor(np.where(label_person1 < 11, 1, 0) )
label_person2 = torch.Tensor(np.where(label_person2 < 11, 1, 0) )
label_person3 = torch.Tensor(np.where(label_person3 < 11, 1, 0) )
label_person4 = torch.Tensor(np.where(label_person4 < 11, 1, 0) )
# (100,)
label_all = torch.cat((label_person1, label_person2, label_person3, label_person4), dim = 0)


# [5, 4, ...] : 每个人的数据均分为 5 份，每次选一份作为测试集
# Useage :  train_input, test_input, train_label, test_label = rawTrace_folds1[i]
rawTrace_folds1 = Get_KFold(rawTrace_person1, label_person1, 5)
rawTrace_folds2 = Get_KFold(rawTrace_person2, label_person2, 5)
rawTrace_folds3 = Get_KFold(rawTrace_person3, label_person3, 5)
rawTrace_folds4 = Get_KFold(rawTrace_person4, label_person4, 5)

# indices = torch.randperm(rawTrace_person_all.size(0))
# rawTrace_person_all = rawTrace_person_all[indices]
# label_all = label_all[indices]
rawTrace_folds_all = Get_KFold(rawTrace_person_all, label_all, 5)

OS_folds1 = Get_KFold(OS_person1, label_person1, 5)
OS_folds2 = Get_KFold(OS_person2, label_person2, 5)
OS_folds3 = Get_KFold(OS_person3, label_person3, 5)
OS_folds4 = Get_KFold(OS_person4, label_person4, 5)

OS_folds_all = Get_KFold(OS_person_all, label_all, 5)

RawTrace_Fold = [rawTrace_folds1, rawTrace_folds2, rawTrace_folds3, rawTrace_folds4]


# for person in range(4):
#     accuracies = []
#     for fold in range(5):
#         train_input, test_input, train_label, test_label = RawTrace_Fold[person][fold]
        
#         # batch norm
#         train_input = batch_norm_1d(train_input).detach()
#         test_input = batch_norm_1d(test_input).detach()
        
#         train_input = train_input.reshape(train_input.size(0), -1)  # 将 train_input 转换为二维形式
#         test_input = test_input.reshape(test_input.size(0), -1)  # 将 test_input 转换为二维形式
        
#         # PCA降维
#         # n_components = test_input.size(0)
#         # pca = PCA(n_components=n_components)
#         # train_input = pca.fit_transform(train_input)
#         # test_input = pca.fit_transform(test_input)
        
#         # 训练SVM模型
#         svm = SVC(kernel='linear')
#         svm.fit(train_input, train_label)
        
#         # 预测测试集标签
#         pred_label = svm.predict(test_input)
        
#         # 计算准确性
#         accuracy = accuracy_score(test_label, pred_label)
#         accuracies.append(accuracy)

#         # 输出每折的准确性和平均准确性
#     print("person{}: ".format(person+1))
#     print("Cross-validation accuracies: {}".format(accuracies))
#     print("Mean accuracy: {}\n".format(np.mean(accuracies)))
    
accuracies = []
for fold in range(5):
    train_input, test_input, train_label, test_label = OS_folds_all[fold]
    
    # batch norm
    train_input = batch_norm_2d(train_input).detach()
    test_input = batch_norm_2d(test_input).detach()
    
    train_input = train_input.reshape(train_input.size(0), -1)  # 将 train_input 转换为二维形式
    test_input = test_input.reshape(test_input.size(0), -1)  # 将 test_input 转换为二维形式
    
    #PCA降维
    # n_components = test_input.size(0)
    # pca = PCA(n_components=n_components)
    # train_input = pca.fit_transform(train_input)
    # test_input = pca.fit_transform(test_input)
    
    # 训练SVM模型
    svm = SVC(kernel='linear', C = 1)
    svm.fit(train_input, train_label)
    
    # 预测测试集标签
    pred_label = svm.predict(test_input)
    
    # 计算准确性
    accuracy = accuracy_score(test_label, pred_label)
    accuracies.append(accuracy)

# 输出每折的准确性和平均准确性
# print("person{}: ".format(person+1))
print("Cross-validation accuracies: {}".format(accuracies))
print("Mean accuracy: {}\n".format(np.mean(accuracies)))