import torch.nn as nn
from torchvision import models 


class ResNet(nn.Module):

  def __init__(self, architecture="resnet18", use_projection=False):
    super(ResNet,self).__init__()
    model = models.resnet18(pretrained = True)
    self.use_projection = use_projection
    self.conv1 = model.conv1
    self.bn1 = model.bn1
    self.relu = model.relu
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4
    self.avgpool = model.avgpool

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x_1=x
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x, x_1
    

class ObjectClassifier(nn.Module):

  def __init__(self, input_dim, class_num=51, extract=True, dropout=0.5):
    super(ObjectClassifier, self).__init__()
    self.linear = nn.Linear(input_dim, 1000)
    self.fc1 = nn.Sequential(nn.Linear(input_dim, 1000), nn.BatchNorm1d(1000, affine=True), 
                             nn.ReLU(inplace=True), nn.Dropout(p=dropout))
    self.fc2 = nn.Linear(1000, class_num)
    self.extract = extract
    self.dropout = dropout

  def forward(self, x):
    emb = self.fc1(x)
    logit = self.fc2(emb)
    if self.extract:
      return logit, emb
    return logit

class PermClassifier(nn.Module):
  def __init__(self, input_dim, projection_dim=100, class_num=9):
    super(PermClassifier, self).__init__()
    self.input_dim = input_dim
    self.projection_dim = projection_dim
    self.conv1 = nn.Sequential(nn.Conv2d(self.input_dim, self.projection_dim, [1,1], stride=[1,1]),
                               nn.BatchNorm2d(self.projection_dim), nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(nn.Conv2d(self.projection_dim, self.projection_dim, [3,3], [2,2]),
                               nn.BatchNorm2d(self.projection_dim), nn.ReLU(inplace=True))
    self.fc1 = nn.Sequential(nn.Linear(self.projection_dim*3*3, self.projection_dim),
                             nn.BatchNorm1d(self.projection_dim, affine=True), nn.ReLU(inplace=True),
                             nn.Dropout(0.5))
    self.fc2 = nn.Linear(self.projection_dim, class_num)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

class RotationClassifier(nn.Module):
  def __init__(self, input_dim, projection_dim=100, class_num=4):
    super(RotationClassifier, self).__init__()
    self.input_dim = input_dim
    self.projection_dim = projection_dim
    self.conv1 = nn.Sequential(nn.Conv2d(self.input_dim, self.projection_dim, [1,1], stride=[1,1]),
                               nn.BatchNorm2d(self.projection_dim), nn.ReLU(inplace=True))
    self.conv2 = nn.Sequential(nn.Conv2d(self.projection_dim, self.projection_dim, [3,3], [2,2]),
                               nn.BatchNorm2d(self.projection_dim), nn.ReLU(inplace=True))
    self.fc1 = nn.Sequential(nn.Linear(self.projection_dim*3*3, self.projection_dim),
                             nn.BatchNorm1d(self.projection_dim, affine=True), nn.ReLU(inplace=True),
                             nn.Dropout(0.5))
    self.fc2 = nn.Linear(self.projection_dim, class_num)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x