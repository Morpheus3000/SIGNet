import os
import importlib.util
import glob

from tqdm import tqdm
import numpy as np
import imageio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Utils import mor_utils


np.seterr(divide='ignore', invalid='ignore')
torch.backends.cudnn.benchmark = True

cudaDevice = ''

if len(cudaDevice) < 1:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('[*] GPU Device selected as default execution device.')
    else:
        device = torch.device('cpu')
        print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
              'Execution maybe slow!')
else:
    device = torch.device('cuda:%s' % cudaDevice)
    print('[*] GPU Device %s selected as default execution device.' %
          cudaDevice)

try:
    cluster_id = '(%s)' % os.environ['CLUSTER_ID_UVA']
except Exception as _:
    cluster_id = '(ENV-UNKNOWN)'

network_script = 'Network.py'
dataloader_script = 'DataloaderMask2Former.py'
net_name = 'NonScaleHomogeneityEdgeBased'

# Change to the model dir.
root_dir = './models/'

# Change to desired folder name
save_dir = './results/%s_%s.png'
os.makedirs(save_dir, exist_ok=True)

finetunedir = root_dir + 'model_release_snapshot.t7'

# Change to the folder with the images
data_root = 'test_imgs/'
img_list = glob.glob(data_root + '*.png')

# Change to the segmentation output folder
seg_fold = 'segmentation_outputs/'

test_list = [x.split('.')[0] for x in img_list]

done = u'\u2713'

net_spec = importlib.util.spec_from_file_location('Network',
                                                  './' + network_script)

dataloader_spec = importlib.util.spec_from_file_location('Dataload_module',
                                                         './' +
                                                         dataloader_script)

net_module = importlib.util.module_from_spec(net_spec)
net_spec.loader.exec_module(net_module)

dataloader_module = importlib.util.module_from_spec(dataloader_spec)
dataloader_spec.loader.exec_module(dataloader_module)

Datapipeline = getattr(dataloader_module, 'GeneralDataset')
Network = getattr(net_module, net_name)

print('[I] STATUS: Create utils instances...', end='')
support = mor_utils(device)
print(done)

print('[I] STATUS: Load network and transfer to device...', end='')
net = Network().to(device)
net, _, _ = support.loadModels(net, finetunedir)
net.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!...", end='')
    net = nn.DataParallel(net)
net.to(device)
print(done)

print('[I] STATUS: Initiate Dataloaders...')
testset = Datapipeline(test_list, data_root)
testLoader = DataLoader(testset, batch_size=1, shuffle=False,
                        num_workers=1, pin_memory=True)
batches_test = len(testLoader)
samples_test = len(testLoader.dataset)
print('\t[*] Test set with %d samples and %d batches.' % (samples_test,
                                                          batches_test),
      end='')
print(done)


global iter_count

iter_count = 0
print('[*] Beginning Inference:')
print('\tFinetune Model from: ', finetunedir)


if __name__ == '__main__':

    t = tqdm(enumerate(testLoader), total=batches_test, leave=False)

    net.eval()
    cnt = 0
    for i, data in t:
        images, file_name = data

        img = Variable(images['rgb']).to(device)[:, :3, :, :]
        colour_sem_img = Variable(images['segs']).to(device)
        raw_sem_img = Variable(images['raw_sem']).to(device)
        inv_img = Variable(images['inv_img']).to(device)
        shd_est_img = Variable(images['shd_est']).to(device)

        pred = net(img, inv_img, colour_sem_img, shd_est_img)
        b, _, _, _ = img.shape

        cnt += 1
        for j in range(b):
            img_gt = img[j].cpu().detach().clone().numpy()
            img_gt = (img_gt / img_gt.max()) * 255
            img_gt = img_gt.transpose((1, 2, 0))
            img_gt = img_gt.astype(np.uint8)

            alb_pred = pred['reflectance'][j].cpu().detach().clone().numpy()
            alb_pred = (alb_pred / alb_pred.max()) * 255
            alb_pred = alb_pred.transpose((1, 2, 0))
            alb_pred = alb_pred.astype(np.uint8)

            shd_pred = pred['shading'][j].cpu().detach().clone().expand(3, 256,
                                                                        256).numpy()
            shd_pred = (shd_pred / shd_pred.max()) * 255
            shd_pred = shd_pred.transpose((1, 2, 0))
            shd_pred = shd_pred.astype(np.uint8)

            imageio.imwrite(save_dir % (file_name[j],
                                        '_img'), img_gt)
            imageio.imwrite(save_dir % (file_name[j],
                                        'pred_alb'), alb_pred)
            imageio.imwrite(save_dir % (file_name[j],
                                        'pred_shd'), shd_pred)


