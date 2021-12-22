import os.path
import glob
from .listdataset import ListDataset
from .util import split2list



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

    return split2list(images, split, default_split=1.0)


def deep_vedio(train_set, test_set, transform=None, target_transform=None,
                  co_transform=None, split=None):
    #train_list, test_list = make_dataset(root,split)

    train_dataset = ListDataset(train_set, transform, target_transform, co_transform)
    test_dataset = ListDataset(test_set, transform, target_transform)

    return train_dataset, test_dataset
