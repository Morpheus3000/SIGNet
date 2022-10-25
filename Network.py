import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
import time
from torchsummary import summary


def printTensorList(data):
    if isinstance(data, dict):
        print('Dictionary Containing: ')
        print('{')
        for key, tensor in data.items():
            print('\t', key, end='')
            print(' with Tensor of Size: ', tensor.size())
            if tensor.dtype != torch.uint8:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
            else:
                print('\t\tMin: %0.4f, Max: %0.4f' % (tensor.min(),
                                                      tensor.max()))

        print('}')
    else:
        print('List Containing: ')
        print('[')
        for tensor in data:
            print('\tTensor of Size: ', tensor.size())
            if tensor.dtype != torch.uint8:
                print('\t\tMin: %0.4f, Mean: %0.4f, Max: %0.4f' % (tensor.min(),
                                                                   tensor.mean(),
                                                                   tensor.max()))
            else:
                print('\t\tMin: %0.4f, Max: %0.4f' % (tensor.min(),
                                                      tensor.max()))

        print(']')


def tdu_palette():
    PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
               [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
               [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
               [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0],
               [64, 64, 128], [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64], [0, 128, 64],
               [128, 128, 64], [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64], [192, 0, 64],
               [64, 128, 64], [192, 128, 64], [64, 0, 192], [192, 0, 192], [64, 128, 192], [192, 128, 192], [0, 64, 64],
               [128, 64, 64], [0, 192, 64], [128, 192, 64], [0, 64, 192], [128, 64, 192], [0, 192, 192],
               [128, 192, 192], [64, 64, 64], [192, 64, 64], [64, 192, 64], [192, 192, 64], [64, 64, 192],
               [192, 64, 192], [64, 192, 192], [192, 192, 192], [32, 0, 0], [160, 0, 0], [32, 128, 0], [160, 128, 0],
               [32, 0, 128], [160, 0, 128], [32, 128, 128], [160, 128, 128], [96, 0, 0], [224, 0, 0], [96, 128, 0],
               [224, 128, 0], [96, 0, 128], [224, 0, 128], [96, 128, 128], [224, 128, 128], [32, 64, 0], [160, 64, 0],
               [32, 192, 0], [160, 192, 0], [32, 64, 128], [160, 64, 128], [32, 192, 128], [160, 192, 128], [96, 64, 0],
               [224, 64, 0], [96, 192, 0], [224, 192, 0], [96, 64, 128], [224, 64, 128], [96, 192, 128],
               [224, 192, 128], [32, 0, 64], [160, 0, 64], [32, 128, 64], [160, 128, 64], [32, 0, 192], [160, 0, 192],
               [32, 128, 192], [160, 128, 192], [96, 0, 64], [224, 0, 64], [96, 128, 64], [224, 128, 64], [96, 0, 192],
               [224, 0, 192], [96, 128, 192], [224, 128, 192], [32, 64, 64], [160, 64, 64], [32, 192, 64],
               [160, 192, 64], [32, 64, 192], [160, 64, 192], [32, 192, 192], [160, 192, 192], [96, 64, 64]]
    return PALETTE


class BilinearSpatialUpSampling(nn.Module):

    """
    Wrapper layer for the torch functional interpolate. The upsampling is
    deprecated and the functional interfaces are not allowed directly in
    Sequential containers.
    Ref:
    https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    """

    def __init__(self, factor, upsample_mode='bilinear', align_corners=True):
        super(BilinearSpatialUpSampling, self).__init__()
        self.factor = factor
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.SUpSamp = F.interpolate

    def forward(self, x):
        return self.SUpSamp(x, mode=self.upsample_mode,
                            scale_factor=self.factor,
                            align_corners=self.align_corners)


class ratioCals(nn.Module):
    """
    Class to handle all the extra ratio calculations. Exposed as layers to a
    network for future reuse.
    """

    def __init__(self, level1_probThreshold=1e-2, level2_probThreshold=1e-4, log_space=False):
        super(ratioCals, self).__init__()
        self.level1_probThreshold = level1_probThreshold
        self.level2_probThreshold = level2_probThreshold
        self.log_space = log_space
        self.register_buffer('filter', torch.Tensor([[[[0, 1, 0],
                                                     [1, 0, -1],
                                                     [0, -1, 0]]]]))

        self.register_buffer('filter2', torch.Tensor([[[[0, -1, 0],
                                                      [-1, 0, 1],
                                                      [0, 1, 0]]]]))

    def getGaussCrossRatios(self, img):

        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        crossed_img = torch.zeros_like(img)

        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)

        # Red-Green
        filt_r1 = F.conv2d(red_chan, weight=self.filter, padding=1)
        filt_g1 = F.conv2d(green_chan, weight=self.filter2, padding=1)
        filt_rg = filt_r1 + filt_g1
        filt_rg = torch.clamp(filt_rg, -1.0, 1.0)
        filt_rg.squeeze_(1)

        # Green-Blue
        filt_g2 = F.conv2d(green_chan, weight=self.filter, padding=1)
        filt_b1 = F.conv2d(blue_chan, weight=self.filter2, padding=1)
        filt_gb = filt_g2 + filt_b1
        filt_gb = torch.clamp(filt_gb, -1.0, 1.0)
        filt_gb.squeeze_(1)

        # Red-Blue
        filt_r2 = F.conv2d(red_chan, weight=self.filter, padding=1)
        filt_b2 = F.conv2d(blue_chan, weight=self.filter2, padding=1)
        filt_rb = filt_r2 + filt_b2
        filt_rb = torch.clamp(filt_rb, -1.0, 1.0)
        filt_rb.squeeze_(1)

        if self.log_space:
            crossed_img[:, 0, :, :] = filt_rg
            crossed_img[:, 2, :, :] = filt_gb
            crossed_img[:, 1, :, :] = filt_rb
        else:
            crossed_img[:, 0, :, :] = torch.exp(filt_rg)
            crossed_img[:, 1, :, :] = torch.exp(filt_gb)
            crossed_img[:, 2, :, :] = torch.exp(filt_rb)
            crossed_img = crossed_img - 1e-7

        crossed_img[zeroMasks == 1]=0
        return crossed_img

        def forward(self, img):
            shadowMasks = self.intrinsicBordersMasks(img)
            output_dict = {'shadow_regions': shadowMasks}
            return output_dict

    def getGaussColourRatios(self, img):

        zeroMasks = torch.zeros_like(img)
        zeroMasks[img == 0] = 1
        colour_img = torch.zeros_like(img)

        log_img = torch.log(img + 1e-7)

        red_chan = log_img[:, 0, :, :].unsqueeze(1)
        green_chan = log_img[:, 1, :, :].unsqueeze(1)
        blue_chan = log_img[:, 2, :, :].unsqueeze(1)

        # Forward difference
        filt_r1 = F.conv2d(red_chan, weight=self.filter, padding=1)
        filt_g1 = F.conv2d(green_chan, weight=self.filter, padding=1)
        filt_b1 = F.conv2d(blue_chan, weight=self.filter, padding=1)

        # Red Channel
        filt_r = torch.clamp(filt_r1, -1.0, 1.0)
        filt_r.squeeze_(1)

        # Green Channel
        filt_g = torch.clamp(filt_g1, -1.0, 1.0)
        filt_g.squeeze_(1)

        # Blue Channel
        filt_b = torch.clamp(filt_b1, -1.0, 1.0)
        filt_b.squeeze_(1)

        if self.log_space:
            colour_img[:, 0, :, :] = filt_r
            colour_img[:, 2, :, :] = filt_g
            colour_img[:, 1, :, :] = filt_b
        else:
            colour_img[:, 0, :, :] = torch.exp(filt_r)
            colour_img[:, 1, :, :] = torch.exp(filt_g)
            colour_img[:, 2, :, :] = torch.exp(filt_b)
            colour_img = colour_img - 1e-7

        colour_img[zeroMasks == 1]=0
        return colour_img


class gaussCrossRatioCal(ratioCals):
    """
    Class to calculate the cross ratio, using a discrete filter.
    """

    def __init__(self):
        super(gaussCrossRatioCal, self).__init__()

    def forward(self, img):
        crossRatio = self.getGaussCrossRatios(img)
        return {'cross': crossRatio}


class AttentionLayer(nn.Module):

    def __init__(self, learnable=False):
        super(AttentionLayer, self).__init__()
        self.learnable = learnable
        self.makeAttention()

    def makeAttention(self):
        if self.learnable:
            self.learn = nn.Conv2d(3, 3, 3, 1, 1)

        self.sig = nn.Sigmoid()

    def forward(self, left_inp, right_inp):
        # print('Left: ', left_inp.shape)
        # print('Right: ', right_inp.shape)
        sigged = self.sig(left_inp)
        mulled = sigged * right_inp
        attentioned = mulled + right_inp
        return attentioned


class VGGEncoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGEncoderBatchNorm, self).__init__()
        self.makeImgEncoder()
        self.makeCCREncoder()
        self.makeSemanticEncoder()
        self.makeInvariantEncoder()
        self.makeShadingEstEncoder()
        self.crossRatio = gaussCrossRatioCal()
        # self.colourRatio = gaussColourRatioCal()

    def makeImgEncoder(self):
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeCCREncoder(self):
        self.cross0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.cross1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.cross2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.cross3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.cross4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeSemanticEncoder(self):
        self.sem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.sem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.sem2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.sem3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.sem4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeInvariantEncoder(self):
        self.inv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.inv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.inv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.inv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.inv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeShadingEstEncoder(self):
        self.shd_est0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.shd_est1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shd_est2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.shd_est3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shd_est4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def forward(self, img, sem, inv_img, shd_est):
        return_dict = {}
        # Get the cross ratio image from the image and use it as prior
        cross_img = self.crossRatio(img)
        # clipped_cross_img = torch.clamp(cross_img['cross'], 0, 1)

        # # Get the cross ratio image from the image and use it as prior
        # colour_img = self.colourRatio(img)
        # clipped_colour_img = torch.clamp(colour_img['colour'], 0, 1)

        conv00 = self.conv0(img) # 64
        conv10 = self.conv1(conv00) # 128
        conv20 = self.conv2(conv10) # 256
        conv30 = self.conv3(conv20) # 512
        conv40 = self.conv4(conv30) # 512

        return_dict['conv00'] = conv00
        return_dict['conv10'] = conv10
        return_dict['conv20'] = conv20
        return_dict['conv30'] = conv30
        return_dict['conv40'] = conv40

        cross00 = self.cross0(cross_img['cross'])
        # cross00 = self.cross0(clipped_cross_img)
        cross10 = self.cross1(cross00)
        cross20 = self.cross2(cross10)
        cross30 = self.cross3(cross20)
        cross40 = self.cross4(cross30)

        return_dict['cross00'] = cross00
        return_dict['cross10'] = cross10
        return_dict['cross20'] = cross20
        return_dict['cross30'] = cross30
        return_dict['cross40'] = cross40
        return_dict['cross_img'] = cross_img['cross']

        sem00 = self.sem0(sem)
        # sem00 = self.sem0(clipped_sem_img)
        sem10 = self.sem1(sem00)
        sem20 = self.sem2(sem10)
        sem30 = self.sem3(sem20)
        sem40 = self.sem4(sem30)

        return_dict['sem00'] = sem00
        return_dict['sem10'] = sem10
        return_dict['sem20'] = sem20
        return_dict['sem30'] = sem30
        return_dict['sem40'] = sem40

        inv00 = self.inv0(inv_img)
        # inv00 = self.inv0(clipped_inv_img)
        inv10 = self.inv1(inv00)
        inv20 = self.inv2(inv10)
        inv30 = self.inv3(inv20)
        inv40 = self.inv4(inv30)

        return_dict['inv00'] = inv00
        return_dict['inv10'] = inv10
        return_dict['inv20'] = inv20
        return_dict['inv30'] = inv30
        return_dict['inv40'] = inv40

        shd_est00 = self.shd_est0(shd_est)
        # shd_est00 = self.shd_est0(clipped_shd_est_img)
        shd_est10 = self.shd_est1(shd_est00)
        shd_est20 = self.shd_est2(shd_est10)
        shd_est30 = self.shd_est3(shd_est20)
        shd_est40 = self.shd_est4(shd_est30)

        return_dict['shd_est00'] = shd_est00
        return_dict['shd_est10'] = shd_est10
        return_dict['shd_est20'] = shd_est20
        return_dict['shd_est30'] = shd_est30
        return_dict['shd_est40'] = shd_est40

        return return_dict


class VGGScaleClampEdgeDecoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGScaleClampEdgeDecoderBatchNorm, self).__init__()
        self.makeLinkedEdgeDecoder()

    def makeLinkedEdgeDecoder(self):
        self.edge_deconvs0 = nn.Sequential(
            nn.ConvTranspose2d(512 * 3, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.edge_deconvs1 = nn.Sequential(
            nn.ConvTranspose2d(512 * 4, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.edge_deconvs2 = nn.Sequential(
            nn.ConvTranspose2d(512 + (256 * 3), 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.edge_deconvs3 = nn.Sequential(
            nn.ConvTranspose2d(256 + (128 * 3), 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.edge_output = nn.Sequential(
            nn.Conv2d(128 + (64 * 3), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.edge_side_output_1 = nn.Sequential(
            nn.Conv2d(512, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.edge_side_output_2 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

        self.edge_side_output_3 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def forward(self, encoderDict):

        edge_dict = {}
        # mid_comb = torch.cat([encoderDict['conv40']], 1)
        mid_comb = torch.cat([encoderDict['conv40'], encoderDict['cross40'],
                              encoderDict['sem40']], 1)

        edge_deconvs0_list = [self.edge_deconvs0(mid_comb)]
        edge_deconvs0_list.append(encoderDict['conv30'])
        edge_deconvs0_list.append(encoderDict['cross30'])
        edge_deconvs0_list.append(encoderDict['sem30'])
        edge_deconvs0_comb = torch.cat(edge_deconvs0_list, 1)

        edge_deconvs1_list = [self.edge_deconvs1(edge_deconvs0_comb)]
        edge_deconvs1_list.append(encoderDict['conv20'])
        edge_deconvs1_list.append(encoderDict['cross20'])
        edge_deconvs1_list.append(encoderDict['sem20'])
        edge_deconvs1_comb = torch.cat(edge_deconvs1_list, 1)

        edge_deconvs2_list = [self.edge_deconvs2(edge_deconvs1_comb)]
        edge_deconvs2_list.append(encoderDict['conv10'])
        edge_deconvs2_list.append(encoderDict['cross10'])
        edge_deconvs2_list.append(encoderDict['sem10'])
        edge_deconvs2_comb = torch.cat(edge_deconvs2_list, 1)

        edge_deconvs3_list = [self.edge_deconvs3(edge_deconvs2_comb)]
        edge_deconvs3_list.append(encoderDict['conv00'])
        edge_deconvs3_list.append(encoderDict['cross00'])
        edge_deconvs3_list.append(encoderDict['sem00'])
        edge_deconvs3_comb = torch.cat(edge_deconvs3_list, 1)

        edge_output_list = [self.edge_output(edge_deconvs3_comb)]



        # Side output generation and clamping
        # Clamping is not needed for illumination edges, because the shadows
        # will generally be weaker activation and we want those, unlike the
        # reflect edges.

        reflec_side_output_1 = self.edge_side_output_1(edge_deconvs1_list[0])
        reflec_side_output_1 = torch.clamp(reflec_side_output_1, 0, 1)
        reflec_side_output_1_m = torch.mean(reflec_side_output_1, dim=1, keepdim=True)
        reflec_side_output_1_m = reflec_side_output_1_m / reflec_side_output_1_m.max()
        reflec_side_output_1_mask = torch.zeros_like(reflec_side_output_1_m)
        reflec_side_output_1_mask[reflec_side_output_1_m > 0.1] = 1
        reflec_side_output_1 = reflec_side_output_1 * reflec_side_output_1_mask
        edge_dict['reflec_edge_side_output1'] = reflec_side_output_1

        reflec_side_output_2 = self.edge_side_output_2(edge_deconvs2_list[0])
        reflec_side_output_2 = torch.clamp(reflec_side_output_2, 0, 1)
        reflec_side_output_2_m = torch.mean(reflec_side_output_2, dim=1, keepdim=True)
        reflec_side_output_2_m = reflec_side_output_2_m / reflec_side_output_2_m.max()
        reflec_side_output_2_mask = torch.zeros_like(reflec_side_output_2_m)
        reflec_side_output_2_mask[reflec_side_output_2_m > 0.1] = 1
        reflec_side_output_2 = reflec_side_output_2 * reflec_side_output_2_mask
        edge_dict['reflec_edge_side_output2'] = reflec_side_output_2

        # Clamp the edge output
        reflec_edge_output = edge_output_list[0]
        reflec_edge_output = torch.clamp(reflec_edge_output, 0, 1)
        reflec_edge_output_m = torch.mean(reflec_edge_output, dim=1, keepdim=True)
        reflec_edge_output_m = reflec_edge_output_m / reflec_edge_output_m.max()
        reflec_edge_output_mask = torch.zeros_like(reflec_edge_output_m)
        reflec_edge_output_mask[reflec_edge_output_m > 0.1] = 1
        reflec_edge_output = reflec_edge_output * reflec_edge_output_mask

        edge_dict['reflect_edge_deconvs0_list'] = edge_deconvs0_list[0]
        edge_dict['reflect_edge_deconvs1_list'] = edge_deconvs1_list[0]
        edge_dict['reflect_edge_deconvs2_list'] = edge_deconvs2_list[0]
        edge_dict['reflect_edge_deconvs3_list'] = edge_deconvs3_list[0]
        edge_dict['reflect_edge_output'] = reflec_edge_output

        return edge_dict


class VGGUnrefinedDecoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGUnrefinedDecoderBatchNorm, self).__init__()
        self.makeUnrefinedReflecDecoder()
        self.makeUnrefinedShadingDecoder()
        self.attention = AttentionLayer()

    def makeUnrefinedReflecDecoder(self):
        self.unrefined_reflec0 = nn.Sequential(
            nn.ConvTranspose2d(512 * 3, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.unrefined_reflec1 = nn.Sequential(
            nn.ConvTranspose2d(512 * 5, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.unrefined_reflec2 = nn.Sequential(
            nn.ConvTranspose2d((512 * 2) + (256 * 3), 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.unrefined_reflec3 = nn.Sequential(
            nn.ConvTranspose2d((256 * 2) + (128 * 3), 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.alb_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 3), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def makeUnrefinedShadingDecoder(self):
        self.unrefined_shd0 = nn.Sequential(
            nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.unrefined_shd1 = nn.Sequential(
            nn.ConvTranspose2d(512 * 4, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.unrefined_shd2 = nn.Sequential(
            nn.ConvTranspose2d((512 * 2) + (256 * 2), 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.unrefined_shd3 = nn.Sequential(
            nn.ConvTranspose2d((256 * 2) + (128 * 2), 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shd_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 2), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

    def forward(self, encoderDict, edgeDict):
        unrefined_dict = {}

        reflec_mid = torch.cat([encoderDict['conv40'], encoderDict['sem40'],
                                encoderDict['inv40']], 1)

        shd_mid = torch.cat([encoderDict['conv40'],
                             encoderDict['shd_est40']], 1)

        unrefined_reflec0_list = []
        store_unrefined_reflec_deconvs0 = self.unrefined_reflec0(reflec_mid)

        unrefined_shd0_list = []
        store_unrefined_shd_deconvs0 = self.unrefined_shd0(shd_mid)

        attn_reflec_edge0 =\
                self.attention(edgeDict['reflect_edge_deconvs0_list'],
                               store_unrefined_reflec_deconvs0)
        unrefined_reflec0_list.append(attn_reflec_edge0)
        unrefined_reflec0_list.append(encoderDict['conv30'])
        unrefined_reflec0_list.append(encoderDict['sem30'])
        unrefined_reflec0_list.append(encoderDict['inv30'])
        unrefined_reflec0_list.append(store_unrefined_shd_deconvs0)
        unrefined_reflec0_comb = torch.cat(unrefined_reflec0_list, 1)

        unrefined_shd0_list.append(store_unrefined_shd_deconvs0)
        unrefined_shd0_list.append(encoderDict['conv30'])
        unrefined_shd0_list.append(encoderDict['shd_est30'])
        unrefined_shd0_list.append(attn_reflec_edge0)
        unrefined_shd0_comb = torch.cat(unrefined_shd0_list, 1)


        unrefined_reflec1_list = []
        store_unrefined_reflec_deconvs1 =\
                self.unrefined_reflec1(unrefined_reflec0_comb)

        unrefined_shd1_list = []
        store_unrefined_shd_deconvs1 =\
                self.unrefined_shd1(unrefined_shd0_comb)

        attn_reflec_edge1 =\
                self.attention(edgeDict['reflect_edge_deconvs1_list'],
                               store_unrefined_reflec_deconvs1)

        unrefined_reflec1_list.append(attn_reflec_edge1)
        unrefined_reflec1_list.append(encoderDict['conv20'])
        unrefined_reflec1_list.append(encoderDict['sem20'])
        unrefined_reflec1_list.append(encoderDict['inv20'])
        unrefined_reflec1_list.append(store_unrefined_shd_deconvs1)
        unrefined_reflec1_comb = torch.cat(unrefined_reflec1_list, 1)

        unrefined_shd1_list.append(store_unrefined_shd_deconvs1)
        unrefined_shd1_list.append(encoderDict['conv20'])
        unrefined_shd1_list.append(encoderDict['shd_est20'])
        unrefined_shd1_list.append(attn_reflec_edge1)
        unrefined_shd1_comb = torch.cat(unrefined_shd1_list, 1)

        unrefined_reflec2_list = []
        store_unrefined_reflec_deconvs2 =\
                self.unrefined_reflec2(unrefined_reflec1_comb)

        unrefined_shd2_list = []
        store_unrefined_shd_deconvs2 =\
                self.unrefined_shd2(unrefined_shd1_comb)

        attn_reflec_edge2 =\
                self.attention(edgeDict['reflect_edge_deconvs2_list'],
                               store_unrefined_reflec_deconvs2)

        unrefined_reflec2_list.append(attn_reflec_edge2)
        unrefined_reflec2_list.append(encoderDict['conv10'])
        unrefined_reflec2_list.append(encoderDict['sem10'])
        unrefined_reflec2_list.append(encoderDict['inv10'])
        unrefined_reflec2_list.append(store_unrefined_shd_deconvs2)
        unrefined_reflec2_comb = torch.cat(unrefined_reflec2_list, 1)

        unrefined_shd2_list.append(store_unrefined_shd_deconvs2)
        unrefined_shd2_list.append(encoderDict['conv10'])
        unrefined_shd2_list.append(encoderDict['shd_est10'])
        unrefined_shd2_list.append(attn_reflec_edge2)
        unrefined_shd2_comb = torch.cat(unrefined_shd2_list, 1)

        unrefined_reflec3_list = []
        store_unrefined_reflec_deconvs3 =\
                self.unrefined_reflec3(unrefined_reflec2_comb)

        unrefined_shd3_list = []
        store_unrefined_shd_deconvs3 =\
                self.unrefined_shd3(unrefined_shd2_comb)

        attn_reflec_edge3 =\
                self.attention(edgeDict['reflect_edge_deconvs3_list'],
                               store_unrefined_reflec_deconvs3)

        unrefined_reflec3_list.append(attn_reflec_edge3)
        unrefined_reflec3_list.append(encoderDict['conv00'])
        unrefined_reflec3_list.append(encoderDict['sem00'])
        unrefined_reflec3_list.append(encoderDict['inv00'])
        unrefined_reflec3_list.append(store_unrefined_shd_deconvs3)
        unrefined_reflec3_comb = torch.cat(unrefined_reflec3_list, 1)

        unrefined_shd3_list.append(store_unrefined_shd_deconvs3)
        unrefined_shd3_list.append(encoderDict['conv00'])
        unrefined_shd3_list.append(encoderDict['shd_est00'])
        unrefined_shd3_list.append(attn_reflec_edge3)
        unrefined_shd3_comb = torch.cat(unrefined_shd3_list, 1)


        store_reflec_output = self.alb_output(unrefined_reflec3_comb)
        attn_unrefined_alb_output =\
                self.attention(edgeDict['reflect_edge_output'],
                               store_reflec_output)
        unrefined_shd_output = self.shd_output(unrefined_shd3_comb)

        unrefined_dict['alb_deconvs0_list'] = unrefined_reflec0_list[0]
        unrefined_dict['alb_deconvs1_list'] = unrefined_reflec1_list[0]
        unrefined_dict['alb_deconvs2_list'] = unrefined_reflec2_list[0]
        unrefined_dict['alb_deconvs3_list'] = unrefined_reflec3_list[0]

        unrefined_dict['shd_deconvs0_list'] = unrefined_shd0_list[0]
        unrefined_dict['shd_deconvs1_list'] = unrefined_shd1_list[0]
        unrefined_dict['shd_deconvs2_list'] = unrefined_shd2_list[0]
        unrefined_dict['shd_deconvs3_list'] = unrefined_shd3_list[0]

        unrefined_dict['alb_output_unrefined'] = attn_unrefined_alb_output
        unrefined_dict['shd_output_unrefined'] = unrefined_shd_output

        return unrefined_dict


class VGGEdgeEncoderBatchNorm(nn.Module):

    def __init__(self):
        super(VGGEdgeEncoderBatchNorm, self).__init__()
        self.makeReflecEdgeEncoder()
        # self.colourRatio = gaussColourRatioCal()

    def makeReflecEdgeEncoder(self):
        self.edge_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.edge_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.edge_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.edge_conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def forward(self, edge):
        return_dict = {}

        edge_conv00 = self.edge_conv0(edge) # 64
        edge_conv10 = self.edge_conv1(edge_conv00) # 128
        edge_conv20 = self.edge_conv2(edge_conv10) # 256
        edge_conv30 = self.edge_conv3(edge_conv20) # 512
        edge_conv40 = self.edge_conv4(edge_conv30) # 512

        return_dict['edge_conv00'] = edge_conv00
        return_dict['edge_conv10'] = edge_conv10
        return_dict['edge_conv20'] = edge_conv20
        return_dict['edge_conv30'] = edge_conv30
        return_dict['edge_conv40'] = edge_conv40

        return return_dict


class VGGDecRefinerEdgeBatchNorm(nn.Module):
    def __init__(self):
        super(VGGDecRefinerEdgeBatchNorm, self).__init__()
        self.makeReflecEncoder()
        self.makeShadingEncoder()
        self.makeReflecDecoder()
        self.makeShadingDecoder()
        self.makeFeatureRecalibrator()
        self.attention = AttentionLayer()

    def makeReflecEncoder(self):
        self.reflec_conv0 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.reflec_conv1 = nn.Sequential(
            nn.Conv2d(64 * 1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.reflec_conv2 = nn.Sequential(
            nn.Conv2d(128 * 1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.reflec_conv3 = nn.Sequential(
            nn.Conv2d(256 * 1, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.reflec_conv4 = nn.Sequential(
            nn.Conv2d(512 * 1, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeShadingEncoder(self):
        self.shading_conv0 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.shading_conv1 = nn.Sequential(
            nn.Conv2d(64 * 1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shading_conv2 = nn.Sequential(
            nn.Conv2d(128 * 1, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.shading_conv3 = nn.Sequential(
            nn.Conv2d(256 * 1, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shading_conv4 = nn.Sequential(
            nn.Conv2d(512 * 1, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

    def makeReflecDecoder(self):
        self.reflec_deconvs0 =  nn.Sequential(
            nn.ConvTranspose2d(512 * 1, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.reflec_deconvs1 = nn.Sequential(
            nn.ConvTranspose2d(512 * 4, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.reflec_deconvs2 = nn.Sequential(
            nn.ConvTranspose2d((512 * 2) + (256 * 2), 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.reflec_deconvs3 = nn.Sequential(
            nn.ConvTranspose2d((256 * 2) + (128 * 2), 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.alb_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 2), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(True)
        )

    def makeShadingDecoder(self):
        self.shading_deconvs0 =  nn.Sequential(
            nn.ConvTranspose2d(512 * 1, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shading_deconvs1 = nn.Sequential(
            nn.ConvTranspose2d(512 * 3, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.shading_deconvs2 = nn.Sequential(
            nn.ConvTranspose2d((512 * 2) + (256 * 1), 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.shading_deconvs3 = nn.Sequential(
            nn.ConvTranspose2d((256 * 2) + (128 * 1), 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.shd_output = nn.Sequential(
            nn.Conv2d((128 * 2) + (64 * 1), 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )

    def makeFeatureRecalibrator(self):
        self.alb_recalibrator = nn.Sequential(
            nn.Conv2d(6, 8, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

    def forward(self, x, x1, edgeEncDict):
        # unrefinedIntrinsics -> List of [Reflec Edge, Unrefined Alb, Illum Edge, Unrefined Shd]
        # Unrefined reflect already has edge attention in it.
        intrinsic_dict = {}
        concat_reflec = torch.cat(x, 1)

        calibrated_reflec = self.alb_recalibrator(concat_reflec)
        # concat_edge_reflec = torch.cat([calibrated_reflec, x1], 1)

        reflec_convs00 = self.reflec_conv0(calibrated_reflec)
        reflec_convs10 = self.reflec_conv1(reflec_convs00)
        reflec_convs20 = self.reflec_conv2(reflec_convs10)
        reflec_convs30 = self.reflec_conv3(reflec_convs20)
        reflec_convs40 = self.reflec_conv4(reflec_convs30)

        shading_convs00 = self.shading_conv0(x1)
        shading_convs10 = self.shading_conv1(shading_convs00)
        shading_convs20 = self.shading_conv2(shading_convs10)
        shading_convs30 = self.shading_conv3(shading_convs20)
        shading_convs40 = self.shading_conv4(shading_convs30)

        # Linked Decoder
        reflec_deconvs0_list = []
        store_reflec_deconvs0 = self.reflec_deconvs0(reflec_convs40)

        shd_deconvs0_list = []
        store_shd_deconvs0 = self.shading_deconvs0(shading_convs40)

        reflec_deconvs0_list.append(store_reflec_deconvs0)
        reflec_deconvs0_list.append(reflec_convs30)
        reflec_deconvs0_list.append(self.attention(edgeEncDict['edge_conv30'],
                                                  reflec_convs30))
        reflec_deconvs0_list.append(store_shd_deconvs0)
        reflec_deconvs0_comb = torch.cat(reflec_deconvs0_list, 1)


        shd_deconvs0_list.append(store_shd_deconvs0)
        shd_deconvs0_list.append(shading_convs30)
        shd_deconvs0_list.append(store_reflec_deconvs0)
        shd_deconvs0_comb = torch.cat(shd_deconvs0_list, 1)

        reflec_deconvs1_list = []
        store_reflec_deconvs1 = self.reflec_deconvs1(reflec_deconvs0_comb)

        shd_deconvs1_list = []
        store_shd_deconvs1 = self.shading_deconvs1(shd_deconvs0_comb)

        reflec_deconvs1_list.append(store_reflec_deconvs1)
        reflec_deconvs1_list.append(reflec_convs20)
        reflec_deconvs1_list.append(self.attention(edgeEncDict['edge_conv20'],
                                                  reflec_convs20))
        reflec_deconvs1_list.append(store_shd_deconvs1)
        reflec_deconvs1_comb = torch.cat(reflec_deconvs1_list, 1)


        shd_deconvs1_list.append(store_shd_deconvs1)
        shd_deconvs1_list.append(shading_convs20)
        shd_deconvs1_list.append(store_reflec_deconvs1)
        shd_deconvs1_comb = torch.cat(shd_deconvs1_list, 1)

        reflec_deconvs2_list = []
        store_reflec_deconvs2 = self.reflec_deconvs2(reflec_deconvs1_comb)

        shd_deconvs2_list = []
        store_shd_deconvs2 = self.shading_deconvs2(shd_deconvs1_comb)

        reflec_deconvs2_list.append(store_reflec_deconvs2)
        reflec_deconvs2_list.append(reflec_convs10)
        reflec_deconvs2_list.append(self.attention(edgeEncDict['edge_conv10'],
                                                 reflec_convs10))
        reflec_deconvs2_list.append(store_shd_deconvs2)
        reflec_deconvs2_comb = torch.cat(reflec_deconvs2_list, 1)


        shd_deconvs2_list.append(store_shd_deconvs2)
        shd_deconvs2_list.append(shading_convs10)
        shd_deconvs2_list.append(store_reflec_deconvs2)
        shd_deconvs2_comb = torch.cat(shd_deconvs2_list, 1)

        reflec_deconvs3_list = []
        store_reflec_deconvs3 = self.reflec_deconvs3(reflec_deconvs2_comb)

        shd_deconvs3_list = []
        store_shd_deconvs3 = self.shading_deconvs3(shd_deconvs2_comb)

        reflec_deconvs3_list.append(store_reflec_deconvs3)
        reflec_deconvs3_list.append(reflec_convs00)
        reflec_deconvs3_list.append(self.attention(edgeEncDict['edge_conv00'],
                                                 reflec_convs00))
        reflec_deconvs3_list.append(store_shd_deconvs3)
        reflec_deconvs3_comb = torch.cat(reflec_deconvs3_list, 1)


        shd_deconvs3_list.append(store_shd_deconvs3)
        shd_deconvs3_list.append(shading_convs00)
        shd_deconvs3_list.append(store_reflec_deconvs3)
        shd_deconvs3_comb = torch.cat(shd_deconvs3_list, 1)

        output_reflec = self.alb_output(reflec_deconvs3_comb)
        output_shd = self.shd_output(shd_deconvs3_comb)

        intrinsic_dict['reflectance'] = output_reflec
        intrinsic_dict['shading'] = output_shd

        return intrinsic_dict


class NonScaleHomogeneityEdgeBased(nn.Module):

    def __init__(self):
        super(NonScaleHomogeneityEdgeBased, self).__init__()
        self.imgEncoder = VGGEncoderBatchNorm()
        self.edgeDecoder = VGGScaleClampEdgeDecoderBatchNorm()
        self.unrefinedDecoder = VGGUnrefinedDecoderBatchNorm()
        self.edgeEncoder = VGGEdgeEncoderBatchNorm()
        self.refinerNet = VGGDecRefinerEdgeBatchNorm()

    def forward(self, img, inv_img, colour_seg, shd_est):
        imgCCREncFeatures = self.imgEncoder(img, colour_seg, inv_img, shd_est)
        edgePrediction = self.edgeDecoder(imgCCREncFeatures)
        unrefinedIntrinsics = self.unrefinedDecoder(imgCCREncFeatures,
                                                        edgePrediction)

        x = [edgePrediction['reflect_edge_output'],
              unrefinedIntrinsics['alb_output_unrefined']]
        x1 = unrefinedIntrinsics['shd_output_unrefined']

        edge_encoder_dict = self.edgeEncoder(
            edgePrediction['reflect_edge_output']
        )

        intrinsics = self.refinerNet(x, x1, edge_encoder_dict)
        output_dict = {}

        output_dict['reflec_edge_64'] = edgePrediction['reflec_edge_side_output1']
        output_dict['reflec_edge_128'] = edgePrediction['reflec_edge_side_output2']
        output_dict['reflec_edge'] = edgePrediction['reflect_edge_output']

        output_dict['unrefined_reflec'] = unrefinedIntrinsics['alb_output_unrefined']
        output_dict['unrefined_shd'] = unrefinedIntrinsics['shd_output_unrefined']

        output_dict['reflectance'] = intrinsics['reflectance']
        output_dict['shading'] = intrinsics['shading']

        output_dict['recon'] = torch.mul(output_dict['reflectance'],
                                         output_dict['shading'])
        output_dict['cross_img'] = imgCCREncFeatures['cross_img']

        return output_dict


if __name__ == '__main__':
    # Minimum run example.

    cudaDevice = ''

    if len(cudaDevice) < 1:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('[*] GPU Device selected as default execution device.')
        else:
            device = torch.device('cpu')
            print('[X] WARN: No GPU Devices found on the system! Using the'
                  ' CPU. Execution maybe slow!')
    else:
        device = torch.device('cuda:%s' % cudaDevice)
        print('[*] GPU Device %s selected as default execution device.' %
              cudaDevice)

    img_batch = torch.Tensor(4, 3, 256, 256).to(device)
    colour_batch = torch.Tensor(4, 3, 256, 256).to(device)
    inv_batch = torch.Tensor(4, 3, 256, 256).to(device)
    shd_est_batch = torch.Tensor(4, 1, 256, 256).to(device)

    net = NonScaleHomogeneityEdgeBased().to(device)

    pred = net(img_batch, inv_batch, colour_batch, shd_est_batch)
    printTensorList({
        'Image': img_batch,
        'Colour Sem': colour_batch,
        'Invariant image': inv_batch
    })
