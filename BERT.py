import scipy.io
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertModel, BertConfig

path = './'

def seed_everything(seed=3407):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

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

# =============================== load data =============================== 
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


# [10, 4, ...] : 每个人的数据均分为 10 份，每次选一份作为测试集
# Useage :  train_input, test_input, train_label, test_label = rawTrace_folds1[i]
rawTrace_folds1 = Get_KFold(rawTrace_person1, label_person1, 10)
rawTrace_folds2 = Get_KFold(rawTrace_person2, label_person2, 10)
rawTrace_folds3 = Get_KFold(rawTrace_person3, label_person3, 10)
rawTrace_folds4 = Get_KFold(rawTrace_person4, label_person4, 10)
rawTrace_folds_all = Get_KFold(rawTrace_person_all, label_all, 10)

OS_folds1 = Get_KFold(OS_person1, label_person1, 10)
OS_folds2 = Get_KFold(OS_person2, label_person2, 10)
OS_folds3 = Get_KFold(OS_person3, label_person3, 10)
OS_folds4 = Get_KFold(OS_person4, label_person4, 10)
OS_folds_all = Get_KFold(OS_person_all, label_all, 10)


rawTrace_folds_all_new = []
for fold in range(10):
    a, b, c, d = rawTrace_folds1[fold]
    
    ta, tb, tc, td = rawTrace_folds2[fold]
    
    a = torch.cat((a, ta), dim = 0)
    b = torch.cat((b, tb), dim = 0)
    c = torch.cat((c, tc), dim = 0)
    d = torch.cat((d, td), dim = 0)
    
    ta, tb, tc, td = rawTrace_folds3[fold]
    a = torch.cat((a, ta), dim = 0)
    b = torch.cat((b, tb), dim = 0)
    c = torch.cat((c, tc), dim = 0)
    d = torch.cat((d, td), dim = 0)
    
    ta, tb, tc, td = rawTrace_folds4[fold]
    a = torch.cat((a, ta), dim = 0)
    b = torch.cat((b, tb), dim = 0)
    c = torch.cat((c, tc), dim = 0)
    d = torch.cat((d, td), dim = 0)
    
    rawTrace_folds_all_new.append((a, b, c, d))

# =============================== load data =============================== 

class TransformerBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, n_heads, n_layers, n_inter):
        super(TransformerBinaryClassifier, self).__init__()
        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            intermediate_size=hidden_size * n_inter,
            max_position_embeddings=2100,
            vocab_size=30522
        )
        
        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=3)
        # self.conv1d2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=3)
        self.dropout = nn.Dropout(p = 0.3)
        
        self.cls = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, x):
        
        x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        is_training = self.training
        if is_training:
            noise = torch.normal(mean=0.0, std=0.2, size=x.size()).to(x.device)
            x = x + noise
        
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # x = self.conv1d2(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x = self.dropout(x)
        # x = self.batch_norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        tmp = self.cls.unsqueeze(0).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((tmp, x), dim = 1)
        
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        # attention_mask = torch.ones(x.shape[:2]).to(x.device)
        # outputs = self.bert(inputs_embeds=x)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits_cls = self.classifier(cls_output)
        
        mean_output = torch.mean(outputs.last_hidden_state, dim=1)
        logits_mean = self.classifier(mean_output)
        
        max_output, _ = torch.max(outputs.last_hidden_state, dim=1)
        logits_max = self.classifier(max_output)
        
        return logits_cls, logits_mean, logits_max



def test(hidden_size,
         num_layers,
         batch_size,
         num_epochs,
         lr,
         wd,
         n_heads,
         n_layers,
         n_inter):
    
    # ======== basic setting ========
    time_steps = 2100
    num_channels = 19
    num_classes = 2
    input_size = num_channels
    # ======== basic setting ========
    
    accuracy_fold = []

    for fold in range(10):
        
        train_input, test_input, train_label, test_label = rawTrace_folds_all[fold]
        train_label = train_label.to(dtype = torch.long)
        test_label = test_label.to(dtype = torch.long)

        dataset = TensorDataset(train_input, train_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = TransformerBinaryClassifier(input_size, hidden_size, num_classes, n_heads, n_layers, n_inter).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        model.train()
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output1, output2, output3 = model(inputs)
                outputs = output1
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {scheduler.get_last_lr()}")
                
                _, predicted = torch.max(outputs.data, dim = 1)
                correct = (predicted == labels).sum().item() / predicted.size(0)
                
                # print(f'Fold [{fold+1}/{10}], Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}, accuracy: {correct:.2f}')
                    
                    
        test_dataset = TensorDataset(test_input, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output1, output2, output3 = model(inputs)
                outputs = output1
                _, predicted = torch.max(outputs.data, dim = 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_fold.append(accuracy)
        # print(f'fold = {fold}: Accuracy of the model on the test set: {accuracy:.2f}%')

    accuracy_final = sum(accuracy_fold) / len(accuracy_fold)
    print(f'average accuracy = {accuracy_final:.2f}%')
    return accuracy_final


# ================= set params ================

hidden_size = 60
num_layers = 1

batch_size = 30
num_epochs = 50

lr = 2e-3
wd = 2e-4

n_layers = 1
n_heads = 1
n_inter = 1

# ================= set params ================


# test_hidden_size = 64

# while test_hidden_size <= 134:
#     result = []
#     for _ in range(10):
#         result.append(test( hidden_size=test_hidden_size,
#                             num_layers=num_layers,
#                             batch_size=batch_size,
#                             num_epochs=num_epochs,
#                             lr=lr,
#                             wd=wd) )  
#     accuracy = sum(result) / len(result)  
#     print(f'test_hidden_size = {test_hidden_size}, accuracy = {accuracy:.2f}%')
    
#     test_hidden_size = test_hidden_size + 10

# for test_layer in range(1, 7):
#     for test_head in range(1, 7):
#         for test_inter in range(1, 5):
#             result = []
#             for _ in range(3):
#                 result.append(test( hidden_size=hidden_size,
#                                     num_layers=num_layers,
#                                     batch_size=batch_size,
#                                     num_epochs=num_epochs,
#                                     lr=lr,
#                                     wd=wd,
#                                     n_heads=test_head,
#                                     n_layers=test_layer,
#                                     n_inter=test_inter) )  
#             accuracy = sum(result) / len(result)  
#             print(f'layer = {test_layer}, head = {test_head}, inter = {test_inter}, accuracy = {accuracy:.2f}%')

test_layer = 3
test_head = 6
test_inter = 2

result = []
for _ in range(10):
    result.append(test( hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        lr=lr,
                        wd=wd,
                        n_heads=test_head,
                        n_layers=test_layer,
                        n_inter=test_inter) )  
accuracy = sum(result) / len(result)  
print(f'{accuracy:.2f}%')