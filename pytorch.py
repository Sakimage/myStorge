#encoding=utf-8
import torch
import csv
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
class mydataset:
    def __init__(self, tens,lab):
        self.dataset =tens
        self.label=lab
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset[idx]
        label = self.label[idx]
        sample = {"text": text, "label": label}
        return sample
class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        #self.pool = nn.AvgPool2d(2, 2)  # 池化层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=4,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU()	
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, 3, 2, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, 3, 2, 1),
            torch.nn.BatchNorm2d(16),
 	    torch.nn.ReLU()
        )
        self.fc1 = nn.Linear(480,250)  # 全连接层
        self.fc2 = nn.Linear(250,120)
        self.fc3 = nn.Linear(120,6)

    def forward(self, x):  # 前向传播

        #x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        #x = self.pool(F.relu(self.conv2(x)))
        x=self.conv1(x)
        x=self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc2(x) # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        x = self.fc3(x)   
 # 从卷基层到全连接层的维度转换
        return x

def trainandsave():
    # 神经网络结构
    net = Net()
      # 学习率为0.001
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(10):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i, data in enumerate(trainloader):  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs= data["text"] # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
    #        print(inputs.shape)
            labels=data["label"]
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable
            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            # forward + backward + optimize
            inputs=inputs.float()
            labels=labels.float()
            outputs = net(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels.long())  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加
            if i % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
    print('Finished Training')
    # 保存神经网络
    return net
if __name__ =='__main__':
    ttens = []
    tlab=[]
    etens = []
    elab=[]
    i=0
    count2=0	
    with open("/home/zc/lyd/train20-40b.csv") as f:
         row=csv.reader(f,delimiter = ',')
         next(row)
         for r in row:
             keyt=int(r[1])
             lab=int(r[2][:-1])
            # if((lab==1)and(count2<=20000)):
            #    count2+=1
            # elif((lab==1)and(count2>20000)):
            #    continue
            # if((lab==1) or (lab==2)):
            #     lab=2
            # if((lab==2)and(count2<=25000)):
            #     count2+=1
            # elif((lab==2)and(count2>25000)):
            #     continue
            # if(lab>4):
            #     continue	 	
             if(keyt==9):
                if(i%5!=0):
                    i=i+1
                    tlab.append(lab)
                    temp=r[0][1:-1].split('|')
                    tempten=[]
                    for j in temp:
                        line=j.split('/')
                        tempten.append(list(map(int,line)))
                    ttens.append(torch.tensor([tempten]))
                else:
                    i=i+1
                    elab.append(lab)
                    temp = r[0][1:-1].split('|')
                    tempten = []
                    for j in temp:
                        line = j.split('/')
                        tempten.append(list(map(int, line)))
                    etens.append(torch.tensor([tempten]))
    td=mydataset(ttens,torch.tensor(tlab))
    test=mydataset(etens,torch.tensor(elab))
    BATCH_SIZE = 8
    trainloader = Data.DataLoader(
        dataset=td,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True
    )
    testloader = Data.DataLoader(
        dataset=test,
        batch_size=BATCH_SIZE,
        num_workers=0,
        drop_last=True
    )

    model = trainandsave()
    accuracy_sum = []
    for i,data in enumerate(testloader):
        test_x = Variable(data["text"]).float()
        test_y = Variable(data["label"]).float()
    #    print(test_x.shape)
        out = model(test_x)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
       # print(torch.max(out,1)[1].numpy(),test_y.numpy())
    print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
