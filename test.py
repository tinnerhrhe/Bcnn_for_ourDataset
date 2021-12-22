import argparse
import os
from os.path import join
import pyexr
import torch
from torch.autograd import Variable
import numpy as np
import models
import pytorch_ssim
from imageio import imsave
import torch.nn.parallel
import torch.backends.cudnn as cudnn
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ssim_loss = pytorch_ssim.SSIM()
l1loss = torch.nn.L1Loss()
def load_data(data):
    min = data.min()
    max = data.max()
    data = torch.FloatTensor(data.size()).copy_(data)
    data.add_(-min).mul_(1.0 / (max - min))
    data = data.mul_(2).add_(-1)
    return data
def save_image(image, filename):
    image = image.add_(1).div_(2)
    image = image.numpy()
    image *= 255.0
    image = image.clip(0, 255)
    #image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".exr"])
def is_exr(filename):
    return any(filename.endswith(extension) for extension in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

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
        #current_frame.describe_channels()
        #current_albedo.describe_channels()
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
        #save_image(gts[0]*255, "./test1.png")
        gts = gts / albedos
        #save_image(gts[0], "./test2.png")
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

    return albedos, directs, normals, depths, gts
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='DeepRendering-implementation')
parser.add_argument('--dataset', required=False, help='unity')
parser.add_argument('--model', type=str, required=True, help='model file')
parser.add_argument('--n_channel_input', type=int, default=3, help='input channel')
parser.add_argument('--n_channel_output', type=int, default=3, help='output channel')
parser.add_argument('--n_generator_filters', type=int, default=64, help="number of generator filters")
parser.add_argument('--arch', '-a', metavar='ARCH', default='runet',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
opt = parser.parse_args()

network_data = torch.load(opt.model)
opt.arch = network_data['arch']
model = models.__dict__[opt.arch](network_data).cuda()
model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True




root_dir = '../DeepIllumination/dataset/test/'
#image_dir = 'dataset/{}/test/albedo'.format(opt.dataset)
image_filenames = [x for x in os.listdir(root_dir) if is_exr(x)]

for image_name in image_filenames:

    albedo, direct, normal, depth, gts = frame_loader(root_dir + image_name)
    '''
    albedo = load_data(torch.from_numpy(albedo).float())
    direct = load_data(torch.from_numpy(direct).float())  # * randScalar
    normal = load_data(torch.from_numpy(normal).float())
    depth = load_data(torch.from_numpy(depth).float())
    gts = load_data(torch.from_numpy(gts).float())  # * randScalar
   
    '''
    #print(albedo,albedo.shape)
    #print(albedo.max(),albedo.min())
    #print("-->>",image_name)
    albedo = torch.from_numpy(albedo).float()
    direct = torch.from_numpy(direct).float() # * randScalar
    normal = torch.from_numpy(normal).float()
    depth = torch.from_numpy(depth).float()
    gts = torch.from_numpy(gts).float()  # * randScalar

    albedo = Variable(albedo).view(1, -1, 576, 1024).cuda()
    direct = Variable(direct).view(1, -1, 576, 1024).cuda()
    normal = Variable(normal).view(1, -1, 576, 1024).cuda()
    depth = Variable(depth).view(1, -1, 576, 1024).cuda()
    gts = Variable(gts).view(1, -1, 576, 1024).cuda()
    output_var=[]
    out = model(torch.cat((direct, normal, depth), 1))
    output_var.append(out)
    output_var = torch.cat(output_var, 1)
    shading_loss = 0
    l1_shading_loss = 0
    laplacian_loss = 0
    # laplacian_loss += l1loss(laplacian_warp(albedo_var[:, frame, :, :, :].cuda() * output_var[:, frame, :, :, :]),
    #                        laplacian_warp(target[:, frame, :, :, :].cuda()))
    l1_shading_loss += l1loss(out[:,:, :, :], gts[:, :, :, :].cuda())
    shading_loss += ssim_loss(out[:,:, :, :], gts[:, :, :, :].cuda())
    print('=> shading_loss: {:.4f} L1_loss: {:.4f}'.format(
        shading_loss,
        l1_shading_loss
    ))
    out = (out * albedo).cpu()
    out_img = out.data[0]
    print(out_img.shape)
    if not os.path.exists("result_bcnn"):
        os.mkdir("result_bcnn")
    if not os.path.exists(os.path.join("result_bcnn", "Final")):
        os.mkdir(os.path.join("result_bcnn", "Final"))
    save_image(out_img, "result_bcnn/Final/{}.png".format(image_name))