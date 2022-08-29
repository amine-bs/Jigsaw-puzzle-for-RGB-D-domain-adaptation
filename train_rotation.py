from dataloader_rotation import MyTransformer, DatasetGeneratorMultimodal
import torch
from torch.utils.data import DataLoader
from networks import ResNet, ObjectClassifier, RotationClassifier
from utils import weights_init, entropy_loss, IteratorWrapper
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

#Parameters
class_num = 51
input_dim = 512
num_workers = 2
epoch = 20
lr = 0.0003
lr_mult = 1
batch_size = 32
weight_decay = 0.005
dropout_p = 0.5
weight_rot = 1
weight_ent = 0.1
device = torch.device('cuda:0')
test_batches = 100
target = "ROD"
source = "synROD"
data_root_source = "/content/output_folder/ROD-synROD/synROD"
train_file_source = "/content/output_folder/ROD-synROD/synROD/synARID_50k-split_sync_train1.txt"
test_file_source = "/content/output_folder/ROD-synROD/synROD/synARID_50k-split_sync_test1.txt"
data_root_target = "/content/output_folder/ROD-synROD/ROD"
train_file_target = "/content/output_folder/ROD-synROD/ROD/wrgbd_40k-split_sync.txt"

#load data
test_transform = MyTransformer([int((256 - 224) / 2), int((256 - 224) / 2)], False)

train_set_source = DatasetGeneratorMultimodal(data_root_source, train_file_source, ds_name=source, do_rot=False)
test_set_source = DatasetGeneratorMultimodal(data_root_source, test_file_source, ds_name=source, do_rot=False, transform=test_transform)
train_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, do_rot=False)
rot_set_source = DatasetGeneratorMultimodal(data_root_source, train_file_source, ds_name=source, do_rot=True)
rot_test_set_source = DatasetGeneratorMultimodal(data_root_source, test_file_source, ds_name=source, do_rot=True)
rot_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, do_rot=True)

train_loader_source = DataLoader(train_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader_source = DataLoader(test_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
train_loader_target = DataLoader(train_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader_target = DataLoader(train_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
rot_source_loader = DataLoader(rot_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
rot_test_source_loader = DataLoader(rot_test_set_source, shuffle=True, batch_size=batch_size, num_workers=num_workers)
rot_target_loader = DataLoader(rot_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)
rot_test_target_loader = DataLoader(rot_set_target, shuffle=True, batch_size=batch_size, num_workers=num_workers)

#Initializing network
netG_rgb = ResNet()
netG_depth = ResNet()
netF = ObjectClassifier(input_dim=input_dim * 2, extract=False)
netF_rot = RotationClassifier(input_dim=input_dim * 2)

netF_rot.apply(weights_init)
netF.apply(weights_init)

netG_rgb = netG_rgb.to(device)
netG_depth = netG_depth.to(device)
netF = netF.to(device)
netF_rot = netF_rot.to(device)

#optimizers
ce_loss=nn.CrossEntropyLoss()

opt_g_rgb = optim.SGD(netG_rgb.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
opt_g_depth = optim.SGD(netG_depth.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
opt_f = optim.SGD(netF.parameters(), lr=lr * lr_mult, momentum=0.9, weight_decay=weight_decay)
opt_f_rot = optim.SGD(netF_rot.parameters(), lr=lr * lr_mult, momentum=0.9, weight_decay=weight_decay)

#training
for epo in range(1,epoch+1):


  print("Epoch {}/{}".format(epo, epoch))
  train_loader_source_rec_iter = train_loader_source
  train_target_loader_iter = IteratorWrapper(train_loader_target)
  test_target_loader_iter = IteratorWrapper(test_loader_target)
  rot_source_loader_iter = IteratorWrapper(rot_source_loader)
  rot_target_loader_iter = IteratorWrapper(rot_target_loader)
  correct = 0
  num_preds = batch_size * len(train_loader_source)

  with tqdm(total=len(train_loader_source), desc="Train") as pb:
    for batch_num, (img_rgb, img_depth, img_label_source) in enumerate(train_loader_source_rec_iter):
      opt_g_rgb.zero_grad(); opt_g_depth.zero_grad(); opt_f.zero_grad(); opt_f_rot.zero_grad()
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

      if weight_rot > 0:
        img_rgb, img_depth, _, rot_label = rot_source_loader_iter.get_next()
        img_rgb = img_rgb.to(device); img_depth = img_depth.to(device); rot_label = rot_label.to(device)

        _, pooled_rgb = netG_rgb(img_rgb)
        _, pooled_depth = netG_depth(img_depth)
        features = torch.cat((pooled_rgb, pooled_depth), 1)
        logits_rot = netF_rot(features)
        loss_rot = ce_loss(logits_rot, rot_label)
        loss = weight_rot * loss_rot
        loss.backward()
        del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss

        img_rgb, img_depth, _, rot_label = rot_target_loader_iter.get_next()
        img_rgb = img_rgb.to(device); img_depth = img_depth.to(device); rot_label = rot_label.to(device)

        _, pooled_rgb = netG_rgb(img_rgb)
        _, pooled_depth = netG_depth(img_depth)
        features = torch.cat((pooled_rgb, pooled_depth), 1)
        logits_rot = netF_rot(features)
        loss_rot = ce_loss(logits_rot, rot_label)
        loss = weight_rot * loss_rot
        loss.backward()
        del img_rgb, img_depth, rot_label, pooled_rgb, pooled_depth, logits_rot, loss
      opt_g_rgb.step(); opt_g_depth.step(); opt_f.step(); opt_f_rot.step()
      pb.update(1)
      
#saving models
models_dir="models"
os.mkdir(models_dir)
torch.save(netG_rgb.state_dict(),models_dir+"/netG_rgb.pth")
torch.save(netG_depth.state_dict(),models_dir+"/netG_depth.pth")
torch.save(netF.state_dict(),models_dir+"/netF.pth")
torch.save(netF_rot.state_dict(),models_dir+"/netF_rot.pth")
      
#Evaluating target recognition
test_set_target = DatasetGeneratorMultimodal(data_root_target, train_file_target, ds_name=target, do_rot=False, transform=test_transform)
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
