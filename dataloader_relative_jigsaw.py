import random
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch
import itertools

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def generate_permutations(seed=300, perm_number=30, grid_size=3*3):
  random.seed(seed)
  grid = list(range(grid_size))
  perms = random.sample(list(itertools.permutations(grid)),29)
  perms.append(tuple(range(9)))
  return perms

permutations = generate_permutations()

def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_image(path):
  img = Image.open(path).convert("RGB")
  return img

def make_dataset(root, label, ds_name):
  images = []
  labelData = open(label)
  for line in labelData:
    data = line.split(' ')
    if not is_image_file(data[0]):
      continue

    
    if ds_name=='synROD':
      path = os.path.join(root, data[0])
      path_rgb=path.replace('***', 'rgb')
      path_depth=path.replace('***', 'depth')

    elif ds_name=='ROD':
      path_rgb = os.path.join(root, "rgb-washington")
      path_depth = os.path.join(root, "surfnorm-washington")
      path_rgb = os.path.join(path_rgb, data[0])
      path_depth = os.path.join(path_depth, data[0])
      path_rgb = path_rgb.replace('***', 'crop')
      path_depth = path_depth.replace('***', 'depthcrop')
    lab = int(data[1])
    images.append((path_rgb, path_depth, lab))

  return images


class MyTransformer():
  def __init__(self, crop, flip):
    self.crop = crop
    self.flip = flip

  def __call__(self, img, perm_index=None):
    img = TF.resize(img, (287,287))
    img = TF.crop(img, self.crop[0], self.crop[1], 255, 255)
    img = TF.to_tensor(img)
    if self.flip :
      img = TF.hflip(img)
    if perm_index is not None:
      perm = [(i + perm_index) % 9 for i in range(9)]
      l=[(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
      tiles = [TF.crop(img, i*85, j*85, 85, 85) for i,j in l]
      tiles_shuffled = [tiles[i] for i in perm]
      img_0 = torch.cat(tiles_shuffled[:3], dim=2)
      img_1 = torch.cat(tiles_shuffled[3:6], dim=2)
      img_2 = torch.cat(tiles_shuffled[6:], dim=2)
      img = torch.cat([img_0, img_1, img_2], dim=1)


    img = TF.resize(img, (224, 224))
    #img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img


class DatasetGeneratorMultimodal(Dataset):
    def __init__(self, root, label, ds_name='synROD', permute=False, transform=None):
        imgs = make_dataset(root, label, ds_name=ds_name)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.permute = permute

    def __getitem__(self, index):
        path_rgb, path_depth, lab = self.imgs[index]
        img_rgb = load_image(path_rgb)
        img_depth = load_image(path_depth)
        perm_index_rgb = None
        perm_index_depth = None

        # If a custom transform is specified apply that transform
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_depth = self.transform(img_depth)
        else:  # Otherwise define a random one (random cropping, random horizontal flip)
            top = random.randint(0, 287 - 255)
            left = random.randint(0, 287 - 255)
            flip = random.choice([True, False])
            if self.permute:
              perm_index_rgb = random.choice(list(range(9)))
              perm_index_depth = random.choice(list(range(9)))


            transform = MyTransformer([top, left], flip)
            # Apply the same transform to both modalities, rotating them if required
            img_rgb = transform(img_rgb, perm_index_rgb)
            img_depth = transform(img_depth, perm_index_depth)

        if self.permute and (self.transform is None):
          perm_lab = (perm_index_rgb - perm_index_depth) % 9
          return img_rgb, img_depth, lab, perm_index_rgb
        return img_rgb, img_depth, lab

    def __len__(self):
        return len(self.imgs)
