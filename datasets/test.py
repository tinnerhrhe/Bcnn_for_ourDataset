import os.path
import glob
import pyexr
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
root = "./train"
def get_flow(filename):
    with open(filename, 'rb') as f:
        #print(np.fromfile(f, np.float32))
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=int(2 * w * h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0], 2))
            # data2D = np.transpose(data2D,[0, 3,1,2])
            return data2D
def make_dataset(rootDir, split=None):
    '''Will search for triplets that go by the pattern '[name].exr  [name]_a.exr    [name].flo' '''
    images = []

    datasets = ['3FO4K2O1B9I5', '3FO4K2OI434X', '3FO4K2OINA7O', '3FO4K2ONCBAQ', '3FO4K2OWMS73', '3FO4K2P802EF', '3FO4K2PDTEIJ' , '3FO4K2PH3QVQ', '3FO4K2PSOO65', '3FO4K2PT3MCB', '3FO4K2PV12YH', '3FO4K2PYHHRS', '3FO4K2VXB3TL', '3FO4K6JA2DRR']

    for dataset in datasets:
        dir = os.path.join(rootDir, dataset, 'result')
        for image in sorted(glob.glob(os.path.join(dir, '*.exr'))):
            if image[-5] == 'a':
                continue

            image_num = image[-14:-4]  #=1
            image_list = []
            flow_list = []
            frames = 2

            for frame in range(frames):
                if frame == 0:
                    current_frame = str(int(image_num)).zfill(10)
                elif frame == 1:
                    current_frame = str(int(image_num)).zfill(10) + 'a'
                if not os.path.isfile(dir + '/' + current_frame + '.exr'):
                    break
                image_list.append(dir + '/' + current_frame + '.exr')

                if os.path.isfile(dir + '/Flow2/' + current_frame + '.flo') and frame < frames - 1:
                    flow_list.append(dir + '/Flow2/' + current_frame + '.flo')

            if len(flow_list) == frames - 1 and len(image_list) == frames:
                images.append([image_list, flow_list])
    features = []
    gts = []
    for image in image_list:
        current_frame = pyexr.open(image)
        current_feature = [current_frame.get("diffuse_dir"), current_frame.get("normal"), current_frame.get("depth"),
                           current_frame.get("albedo")]
        print(current_feature[2].shape)
        current_feature = np.concatenate(current_feature, axis=2)
        current_feature = np.transpose(current_feature, (2, 0, 1))[np.newaxis, :, :, :]
        features.append(current_feature)

        gt_feature = current_frame.get("diffuse")
        #print(gt_feature,gt_feature.shape)
        gt_feature = np.transpose(gt_feature, (2, 0, 1))[np.newaxis, :, :, :]
        gts.append(gt_feature)

    features = np.concatenate(features, axis=0)
    gts = np.concatenate(gts, axis=0)
    #print(len(gts),gts.shape)
    #print(len(features),features.shape)
    flows = []
    for flow in flow_list:
        current_flow = get_flow(flow)[np.newaxis, :, :, :]
        flows.append(current_flow)
    #print(flows[0],flows[0].shape,len(flows))
    flows = np.concatenate(flows, axis=0)
    features = torch.from_numpy(features).float()
    gts = torch.from_numpy(gts).float()
    flows = torch.from_numpy(flows).float()
    img = transforms.ToPILImage()(gts[0])
    img.save(root + "/gts.png")
make_dataset(root)