import os.path
import glob
import pyexr
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
root="file_dir"
image=os.path.join(root, 'test.exr')
current_frame = pyexr.open(image)
print(current_frame)
current_frame.describe_channels() #return channnels
gt_ = current_frame.get("View Layer")
print(gt_,gt_.shape)
gt_rgb=gt_[:,:,0:3]   #return (r,g,b) 3 channel matrix
gt_t = np.transpose(gt_rgb, (2, 0, 1))[np.newaxis, :, :, :] #adjust dimension
gts=[]     #may be more than one exr, then use list
gts.append(gt_t)
gts = np.concatenate(gts, axis=0)
gts = torch.from_numpy(gts).float()
img = transforms.ToPILImage()(gts[0])   #you can choose one to display
#img.save(root + "/gts.png")
img