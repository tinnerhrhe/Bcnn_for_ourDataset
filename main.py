import argparse
import os
import shutil
import time

import datetime
from tensorboardX import SummaryWriter
import numpy as np
from imageio import imsave
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import pytorch_ssim
import flow_transforms
import models
import datasets

from warp import Warp
from smooth import Laplacian_warp

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('data', metavar='DIR',
     #               help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='deep_vedio',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=1.0, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')
parser.add_argument('--arch', '-a', metavar='ARCH', default='runet',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1500, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--milestones', default=[1000,3000,5000, 10000, 20000, 40000, 80000], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

n_iter = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def main():
    global args, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
  

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])
    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])
    root_dir = "../DeepIllumination/dataset/"
    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "val")
    print("=> fetching img pairs in '{}'".format(root_dir))
    train_set, test_set = datasets.__dict__[args.dataset](
        train_dir,
        test_dir,
        transform=input_transform,
        target_transform=target_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                    len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = model.parameters()
   
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)

    #optimizer = torch.optim.Adadelta(param_groups,1.0,0.9)


    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)
        #train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # evaluate on validation set

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict()
        }, False)

        #if epoch % 10 == 0 or epoch == 0:
            #val(epoch+1, val_loader, model)

def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    model.train()
    ssim_loss = pytorch_ssim.SSIM()
    l1loss = torch.nn.L1Loss()

    end = time.time()
    # features   batch * frames * channels(diffuse_dir, normal, depth, albedo) * width * height
    # gts        batch * frames * diffuse * width * height
    # flows      batch * frames * width * height * 2
    for i, (features, gts) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #warp = Warp(flows.shape[0], flows.shape[2], flows.shape[3])
        laplacian_warp = Laplacian_warp(3)

        #flow_var = torch.autograd.Variable(flows.cuda())
        frames = features.shape[1]

        randScalar = torch.ones(3)
        randScalar[int(torch.randint(0,3,(1,))[0])] = torch.rand(1)[0] 
        randScalar = randScalar.view(1,1,3,1,1)

        albedo_feature = features[:,:,9:12,:,:]
        input_feature = [features[:,:,0:3,:,:]  * randScalar, features[:,:,3:6,:,:]]
        depth_feature = features[:,:,6:9,:,:]
        target = gts * randScalar

        input_var = torch.autograd.Variable(torch.cat(input_feature,2))
        target_var = torch.autograd.Variable(target)
        albedo_var = torch.autograd.Variable(albedo_feature)
        depth_var = torch.autograd.Variable(depth_feature.cuda())
        output_var = []

        for frame in range(frames):
            frame_input = input_var[:,frame,:,:,:].cuda()

            output = model(torch.cat((frame_input, depth_var[:,frame,:,:,:]), 1)).unsqueeze(1)
            output_var.append(output)

        output_var = torch.cat(output_var, 1)

        shading_loss = 0
        l1_shading_loss = 0
        laplacian_loss = 0
        for frame in range(frames):
            laplacian_loss += l1loss(laplacian_warp(albedo_var[:,frame,:,:,:].cuda() * output_var[:,frame,:,:,:]), laplacian_warp(target[:,frame,:,:,:].cuda()))
            l1_shading_loss += l1loss( output_var[:,frame,:,:,:], target[:,frame,:,:,:].cuda())
            shading_loss += ssim_loss( output_var[:,frame,:,:,:], target[:,frame,:,:,:].cuda())
        '''
        temporal_loss = 0
        for frame in range(frames-1):
            warp_output1, warp_output2 = warp(output_var[:,frame,:,:,:], output_var[:,frame + 1,:,:,:], flow_var[:,frame,:,:,:])
            warp_res = warp_output2 - warp_output1
            temporal_loss += l1loss(warp_res, torch.autograd.Variable(torch.zeros_like(warp_res.data)))
        '''
        loss = 0.8 * shading_loss  + 0.2 * laplacian_loss
        
        losses.update(loss.data, target.size(0))
        train_writer.add_scalar('ssim_loss', shading_loss.data, n_iter)
        train_writer.add_scalar('l1_loss', l1_shading_loss.data, n_iter)
        #train_writer.add_scalar('temporal_loss', temporal_loss.data, n_iter)
        train_writer.add_scalar('laplacian_loss', laplacian_loss.data, n_iter)
        n_iter += 1

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses))
        if i >= epoch_size:
            break

    return losses.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))
def val(epoch, val_loader, model):
    for i, (features, gts) in enumerate(val_loader):
        frames = features.shape[1]
        albedo_feature = features[:, :, 9:12, :, :]
        input_feature = [features[:, :, 0:3, :, :], features[:, :, 3:6, :, :]]
        depth_feature = features[:, :, 6:9, :, :]
        target = gts

        input_var = torch.autograd.Variable(torch.cat(input_feature, 2))
        target_var = torch.autograd.Variable(target)
        albedo_var = torch.autograd.Variable(albedo_feature)
        depth_var = torch.autograd.Variable(depth_feature.cuda())
        output_var = []

        for frame in range(frames):
            frame_input = input_var[:, frame, :, :, :].cuda()

            output = model(torch.cat((frame_input, depth_var[:, frame, :, :, :]), 1))



        ssim_loss = pytorch_ssim.SSIM()
        l1loss = torch.nn.L1Loss()
        shading_loss = 0
        l1_shading_loss = 0
        laplacian_loss = 0
        for frame in range(frames):
            l1_shading_loss += l1loss(output[:, :, :, :], target[:, frame, :, :, :].cuda())
            shading_loss += ssim_loss(output[:, :, :, :], target[:, frame, :, :, :].cuda())

        print('=> shading_loss: {:.4f} L1_loss: {:.4f}'.format(
            shading_loss,
            l1_shading_loss
        ))
        out = (output[:,:, :, :] * albedo_var[:,:,:,:].cuda()).cpu()
        out_img = out.data[0]
        #print(out_img[0].shape)
        if not os.path.exists("validation"):
            os.mkdir("validation")
        if not os.path.exists(os.path.join("validation", "dataset")):
            os.mkdir(os.path.join("validation", "dataset"))
        save_image(out_img[0],"validation/{}/{}_Fake.png".format("dataset", i))
def save_image(image, filename):
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


if __name__ == '__main__':
    main()
