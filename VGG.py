'''
导入库
'''
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import math
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
from torchvision.transforms import Compose, ToTensor, Resize
import gc
import csv


# 对输入图像进行处理，转换为（224，224）,因为resnet18要求输入为（224，224），并转化为tensor
def input_transform():
    return Compose([
        Resize(224),  # 改变尺寸
        ToTensor(),  # 变成tensor
    ])


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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    # inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,  # 因为mnist为（1，28，28）灰度图，因此输入通道数为1
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # [2, 2, 2, 2]和结构图[]X2是对应的
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:  # 加载模型权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

if __name__ =='__main__':
    ttens = []
    tlab=[]
    etens = []
    elab=[]
    i=0
    count2=0
    count3=0
    count4=0
    count5=0	
    with open("/home/zc/lyd/cnntrain3.csv") as f:
         row=csv.reader(f,delimiter = ',')
         next(row)
         for r in row:
             keyt=int(r[1])
             lab = int(r[2][:-1])
            # if ((lab == 1) and (count2 <= 3000)):
            #     count2 += 1
            # elif ((lab == 1) and (count2 > 3000)):
            #     continue
            # if ((lab == 2) and (count3 <= 3000)):
            #     count3 += 1
            # elif ((lab == 2) and (count3 > 3000)):
            #     continue
            # if ((lab == 3) and (count4 <= 3000)):
            #     count4 += 1
            # elif ((lab == 3) and (count4 > 3000)):
            #     continue
            # if ((lab == 4) and (count5 <= 3000)):
            #     count5 += 1
            # elif ((lab == 4) and (count5 > 3000)):
            #     continue
             if(keyt==16):
                if(i%5!=0):
                    i=i+1
                    tlab.append(int(r[2][:-1]))
                    temp=r[0][1:-1].split('|')
                    tempten=[]
                    for j in temp:
                        line=j.split('/')
                        tempten.append(list(map(int,line)))
                    ttens.append(torch.tensor([tempten]))
                else:
                    i=i+1
                    elab.append(int(r[2][:-1]))
                    temp = r[0][1:-1].split('|')
                    tempten = []
                    for j in temp:
                        line = j.split('/')
                        tempten.append(list(map(int, line)))
                    etens.append(torch.tensor([tempten]))
    td=mydataset(ttens,torch.tensor(tlab))
    test=mydataset(etens,torch.tensor(elab))
    BATCH_SIZE = 1
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
    net = resnet18()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(5):
        for i, data in enumerate(testloader):
            test_x = Variable(data["text"]).float()
            test_y = Variable(data["label"]).float()

            predict = net(test_x)
            loss = loss_func(predict, test_y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 2000 == 0:
                print('epoch:{}, step:{}, loss:{}'.format(epoch,i, loss))
    accuracy_sum = []
    for i,data in enumerate(testloader):
        test_x = Variable(data["text"]).float()
        test_y = Variable(data["label"]).float()
        out = net(test_x)
        # print('test_out:\t',torch.max(out,1)[1])
        # print('test_y:\t',test_y)
        accuracy = torch.max(out,1)[1].numpy() == test_y.numpy()
        accuracy_sum.append(accuracy.mean())
       # print('accuracy:\t',accuracy.mean())
    print('总准确率：\t',sum(accuracy_sum)/len(accuracy_sum))
