import torch.utils.data as data
import os
import os.path
from os import listdir
from os.path import join
#from scipy.ndimage import imread
import pyexr
import numpy as np
import torch
from .util import is_exr
def frame_loader(path):
    normals = []
    depths = []
    directs = []
    albedos = []
    gts = []
    #path = join(path, "path")
    path = [path]
    for image in path:
        #print("---------------",path)
        image_gbuffer = join(image, "gbuffer.exr")
        image_albedo = join(image, "albedo.exr")
        current_frame = pyexr.open(image_gbuffer)
        current_albedo = pyexr.open(image_albedo)
        key = list(current_frame.root_channels)[0]
        gbuffer = current_frame.get(key)
        gt = gbuffer[:, :, 0:3]  # return (r,g,b) 3 channel matrix
        depth_1 = gbuffer[:,:,4].reshape((576, 1024, 1))
        depth = np.concatenate((depth_1, depth_1, depth_1), axis=2)
        direct = gbuffer[:, :, 5:8]
        normal = gbuffer[:, :, -3:]
        albedo_buffer = current_albedo.get(key)
        albedo = albedo_buffer[:, :, 0:3]

        albedo = np.transpose(albedo, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        albedos.append(albedo)
        albedos = np.concatenate(albedos, axis=0)
        gt = np.transpose(gt, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        gts.append(gt)
        gts = np.concatenate(gts, axis=0)
        gts = gts /albedos
        direct = np.transpose(direct, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        directs.append(direct)
        directs = np.concatenate(directs, axis=0)
        directs = directs / albedos
        normal = np.transpose(normal, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        normals.append(normal)
        normals = np.concatenate(normals, axis=0)
        depth = np.transpose(depth, (2, 0, 1))[np.newaxis, :, :, :]  # adjust dimension
        depths.append(depth)
        depths = np.concatenate(depths, axis=0)
        feature = np.concatenate((direct,normal,depth,albedo), axis=1)
    return feature, gts

class ListDataset(data.Dataset):
    def __init__(self, image_dir, transform=None, target_transform=None,
                 co_transform=None, loader=frame_loader):

        #self.root = root
        #self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.root = join(image_dir, "test")
        # print("-------------",image_dir)
        self.image_filenames = [x for x in listdir(self.root) if is_exr(x)]
    def __getitem__(self, index):
        #image_list, flow_list = self.path_list[index]

        features, gts = self.loader(join(self.root, self.image_filenames[index]))
        features = torch.from_numpy(features).float()
        gts = torch.from_numpy(gts).float()


        return  features, gts
    def __len__(self):
        return len(self.image_filenames)
