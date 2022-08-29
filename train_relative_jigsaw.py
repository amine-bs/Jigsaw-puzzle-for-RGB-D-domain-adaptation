from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from dataloader_relative_jigsaw import DatasetGeneratorMultimodal, MyTransformer
from networks import ResNet, ObjectClassifier, PermClassifier
from utils import entropy_loss, weights_init, IteratorWrapper
from torch import nn
from tqdm import tqdm
import os


#Parameters
class_num = 51
input_dim = 512
num_workers = 2
epoch = 5
lr = 0.0003
lr_mult = 1
batch_size = 64
weight_decay = 0.005
dropout_p = 0.5
weight_jigsaw = 0.7
weight_ent = 0.1
device = torch.device('cuda:0')
test_batches = 100
target = "ROD"
source = "synROD"
data_root_source = "/ROD-synROD/synROD"
train_file_source = "/ROD-synROD/synROD/synARID_50k-split_sync_train1.txt"
test_file_source = "/ROD-synROD/synROD/synARID_50k-split_sync_test1.txt"
data_root_target = "/ROD-synROD/ROD"
train_file_target = "/ROD-synROD/ROD/wrgbd_40k-split_sync.txt"

#load data
test_transform = MyTransformer([int((256 - 224) / 2), int((256 - 224) / 2)], False)
train_set_source = DatasetGeneratorMultimodal(data_root_source, train_file_source, ds_name=source, permute=False)
test_set_source = DatasetGeneratorMultimodal(data_root_source, test_file_source, ds_name=source, permute=False, transform=test_transform)
train_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, permute=False)
jigsaw_set_source = DatasetGeneratorMultimodal(data_root_source, train_file_source, ds_name=source, permute=True)
jigsaw_test_set_source = DatasetGeneratorMultimodal(data_root_source, test_file_source, ds_name=source, permute=True)
jigsaw_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, permute=True)

train_loader_source = DataLoader(train_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader_source = DataLoader(test_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
train_loader_target = DataLoader(train_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader_target = DataLoader(train_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
jigsaw_source_loader = DataLoader(jigsaw_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
jigsaw_test_source_loader = DataLoader(jigsaw_test_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
jigsaw_target_loader = DataLoader(jigsaw_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
jigsaw_test_target_loader = DataLoader(jigsaw_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)

#initializing network
netG_rgb = ResNet()
netG_depth = ResNet()
netF = ObjectClassifier(input_dim=input_dim * 2, extract=False)
netF_jigsaw = PermClassifier(input_dim=input_dim * 2, class_num=9)

netF_jigsaw.apply(weights_init)
netF.apply(weights_init)

netG_rgb = netG_rgb.to(device)
netG_depth = netG_depth.to(device)
netF = netF.to(device)
netF_jigsaw = netF_jigsaw.to(device)

#optimizers
ce_loss=nn.CrossEntropyLoss()
opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
opt_g_depth = optim.SGD(netG_depth.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=lr * lr_mult, momentum=0.9, weight_decay=weight_decay)
opt_f_jigsaw = optim.SGD(netF_jigsaw.parameters(), lr=lr * lr_mult, momentum=0.9, weight_decay=weight_decay)

#training
for epo in range(1,epoch+1):

  print("Epoch {}/{}".format(epo, epoch))
  train_loader_source_rec_iter = train_loader_source
  train_target_loader_iter = IteratorWrapper(train_loader_target)
  test_target_loader_iter = IteratorWrapper(test_loader_target)
  jigsaw_source_loader_iter = IteratorWrapper(jigsaw_source_loader)
  jigsaw_target_loader_iter = IteratorWrapper(jigsaw_target_loader)
  correct = 0
  num_preds = batch_size * len(train_loader_source)

  with tqdm(total=len(train_loader_source), desc="Train") as pb:
    for batch_num, (img_rgb, img_depth, img_label_source) in enumerate(train_loader_source_rec_iter):
      opt_g_rgb.zero_grad(); opt_g_depth.zero_grad(); opt_f.zero_grad(); opt_f_jigsaw.zero_grad()
      img_rgb=img_rgb.to(device); img_depth=img_depth.to(device); img_label_source=img_label_source.to(device)
      feat_rgb, _ = netG_rgb(img_rgb)
      feat_depth, _ = netG_depth(img_depth)
      features_source = torch.cat((feat_rgb, feat_depth), 1)
      logits = netF(features_source)
      loss_rec = ce_loss(logits, img_label_source)
      correct += (torch.argmax(logits, dim=1)==img_label_source).sum().item()

      if weight_ent > 0:
        img_rgb, img_depth, _ = train_target_loader_iter.get_next()
        img_rgb = img_rgb.to(device); img_depth = img_depth.to(device)
        feat_rgb, _ = netG_rgb(img_rgb)
        feat_depth, _ = netG_depth(img_depth)
        features_target = torch.cat((feat_rgb, feat_depth), 1)
        logits = netF(features_target)
        loss_ent = entropy_loss(logits)
      else:
        loss_ent = 0
        
      loss = loss_rec + weight_ent * loss_ent
      loss.backward()
      del img_rgb, img_depth, img_label_source, feat_rgb, feat_depth, logits

      if weight_jigsaw > 0:
        img_rgb, img_depth, _, jigsaw_label = jigsaw_source_loader_iter.get_next()
        img_rgb = img_rgb.to(device); img_depth = img_depth.to(device); jigsaw_label = jigsaw_label.to(device)

        _, pooled_rgb = netG_rgb(img_rgb)
        _, pooled_depth = netG_depth(img_depth)
        features = torch.cat((pooled_rgb, pooled_depth), 1)
        logits_jigsaw = netF_jigsaw(features)
        loss_jigsaw = ce_loss(logits_jigsaw, jigsaw_label)
        loss = weight_jigsaw * loss_jigsaw
        loss.backward()
        del img_rgb, img_depth, jigsaw_label, pooled_rgb, pooled_depth, logits_jigsaw, loss

        img_rgb, img_depth, _, jigsaw_label = jigsaw_target_loader_iter.get_next()
        img_rgb = img_rgb.to(device); img_depth = img_depth.to(device); jigsaw_label = jigsaw_label.to(device)

        _, pooled_rgb = netG_rgb(img_rgb)
        _, pooled_depth = netG_depth(img_depth)
        features = torch.cat((pooled_rgb, pooled_depth), 1)
        logits_jigsaw = netF_jigsaw(features)
        loss_jigsaw = ce_loss(logits_jigsaw, jigsaw_label)
        loss = weight_jigsaw * loss_jigsaw
        loss.backward()
        del img_rgb, img_depth, jigsaw_label, pooled_rgb, pooled_depth, logits_jigsaw, loss
      opt_g_rgb.step(); opt_g_depth.step(); opt_f.step(); opt_f_jigsaw.step()
      pb.update(1)

#saving models
models_dir = "models"
os.mkdir(models_dir)
torch.save(netG_rgb.state_dict(),models_dir+"/netG_rgb.pth")
torch.save(netG_depth.state_dict(),models_dir+"/netG_depth.pth")
torch.save(netF.state_dict(),models_dir+"/netF.pth")
torch.save(netF_jigsaw.state_dict(),models_dir+"/netF_jigsaw.pth")

#Evaluating target object recognition
test_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, permute=False, transform=test_transform)
correct, num_predictions = 0, 0
with tqdm(total=len(test_loader_target), desc="Test") as pb:
  torch.set_grad_enabled(False)
  netF.eval(); netG_depth.eval(); netG_rgb.eval()
  test_target_loader_iter = iter(test_loader_target)
  for num_batches, (img_rgb, img_depth, img_label) in enumerate (test_target_loader_iter):
    img_rgb = img_rgb.to(device); img_depth = img_depth.to(device); img_label = img_label.to(device)
    feat_rgb, _ = netG_rgb(img_rgb)
    feat_depth, _ = netG_depth(img_depth)
    features = torch.cat((feat_rgb, feat_depth), 1)
    preds = netF(features)
    correct += (torch.argmax(preds, dim=1)==img_label).sum().item()
    num_predictions += preds.shape[0]
    pb.update(1)
  val_acc = correct / num_predictions
print(val_acc)