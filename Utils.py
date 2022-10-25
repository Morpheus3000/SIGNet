import os
import imageio
import numpy as np
import torch
import torch.nn as nn


class mor_utils:

    def __init__(self, device):
        self.device = device

    def printTensorList(self, data):
        if isinstance(data, dict):
            print('Dictionary Containing: ')
            print('{')
            for key, tensor in data.items():
                print('\t', key, end='')
                print(' with Tensor of Size: ', tensor.size())
            print('}')
        else:
            print('List Containing: ')
            print('[')
            for tensor in data:
                print('\tTensor of Size: ', tensor.size())
            print(']')

    def saveModels(self, model, optims, iterations, path):
        # cpu = torch.device('cpu')
        if isinstance(model, nn.DataParallel):
            checkpoint = {
                'iters': iterations,
                'model': model.module.state_dict(),
                # 'model': model.module.to(cpu).state_dict(),
                'optimizer': optims.state_dict()
            }
        else:
            checkpoint = {
                'iters': iterations,
                'model': model.state_dict(),
                # 'model': model.to(cpu).state_dict(),
                'optimizer': optims.state_dict()
            }
        torch.save(checkpoint, path)
        # model.to(self.device)

    def loadModels(self, model, path, optims=None, Test=True):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        if not Test:
            optims.load_state_dict(checkpoint['optimizer'])
        return model, optims, checkpoint['iters']

    def dumpOutputs(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)
            imageio.imwrite((vis + '/%s_pred_shd.png') % filename, pred_s)

    def dumpOutputs2(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            pred_e = preds[2].cpu().detach().clone().numpy()
            pred_e[pred_e < 0] = 0
            pred_e = (pred_e / pred_e.max()) * 255
            pred_e = pred_e.transpose((1, 2, 0))
            pred_e = pred_e.astype(np.uint8)

            pred_u = preds[3].cpu().detach().clone().numpy()
            pred_u[pred_u < 0] = 0
            pred_u = (pred_u / pred_u.max()) * 255
            pred_u = pred_u.transpose((1, 2, 0))
            pred_u = pred_u.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)
            imageio.imwrite((vis + '/%s_pred_shd.png') % filename, pred_s)
            imageio.imwrite((vis + '/%s_pred_edge.png') % filename, pred_e)
            imageio.imwrite((vis + '/%s_pred_unrefined.png') % filename, pred_u)

    def dumpOutputs3(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            pred_s = preds[1].cpu().detach().clone().numpy()
            pred_s[pred_s < 0] = 0
            pred_s = (pred_s / pred_s.max()) * 255
            pred_s = pred_s.transpose((1, 2, 0))
            pred_s = pred_s.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            shd = gts[2].cpu().detach().clone().numpy() * 255
            shd = shd.astype(np.uint8)
            shd = shd.transpose(1, 2, 0)

            norm = preds[2].cpu().detach().clone().numpy() * 255
            norm[norm < 0] = 0
            norm = (norm / norm.max()) * 255
            norm = norm.astype(np.uint8)
            norm = norm.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, shd), axis=1)
            row2 = np.concatenate((norm, pred_a, pred_s), axis=1)
            full = np.concatenate((row1, row2), axis=0)

            imageio.imwrite(vis + '/' + filename % (num, iteration), full)

        else:
            for k, ele in preds.items():
                pred = ele.cpu().detach().clone().numpy()
                pred[pred < 0] = 0
                pred = (pred / pred.max()) * 255
                pred = pred.transpose((1, 2, 0))
                pred = pred.astype(np.uint8)
                imageio.imwrite((vis + '/%s_%s.png') % (filename, k), pred)

    def dumpReflec(self, vis, preds, gts=None, num=13, iteration=0,
                    filename='Out_%d_%d.png', Train=True):

        if Train:
            """Function to Collage the predictions with the outputs. Expects a single
            set and not batches."""

            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            img = gts[0].cpu().detach().clone().numpy() * 255
            img = img.astype(np.uint8)
            img = img.transpose(1, 2, 0)

            alb = gts[1].cpu().detach().clone().numpy() * 255
            alb = alb.astype(np.uint8)
            alb = alb.transpose(1, 2, 0)

            row1 = np.concatenate((img, alb, pred_a), axis=1)

            imageio.imwrite(vis + '/' + filename % (num, iteration), row1)

        else:
            pred_a = preds[0].cpu().detach().clone().numpy()
            pred_a = (pred_a / pred_a.max()) * 255
            pred_a = pred_a.transpose((1, 2, 0))
            pred_a = pred_a.astype(np.uint8)

            imageio.imwrite((vis + '/%s_pred_alb.png') % filename, pred_a)

    def run_preflight_tests(self, model, dataset_dict, model_dict):
        done = u'\u2713'
        print('\t[I] STATUS: Test network forward systems...', end='')
        with torch.no_grad():
            pred = model(dataset_dict['inputs'])
        print(done)
        print('\t[I] STATUS: Sanity check on the network predictions')
        self.printTensorList(pred)
        _, _, w, h = dataset_dict['inputs'].shape
        print('\t[I] STATUS: Test image dump systems...', end='')
        self.dumpOutputs(dataset_dict['dest'], [pred['reflectance'][0, :, :, :],
                                                pred['shading'][0, :, :, :].expand(3, w, h),
                                                pred['edge'][0, :, :, :]],
                         gts=[dataset_dict['inputs'][0, :, :, :],
                              dataset_dict['inputs'][0, :, :, :],
                              dataset_dict['inputs'][0, :, :, :]],
                         num=13,
                         iteration=1337)
        print(done)

        print('\t[I] STATUS: Test model saving systems...', end='')
        self.saveModels(model, model_dict['optimizer'], 1337,
                        model_dict['dest'] + '/test_dump.t7')
        print(done)
        print('\t[I] STATUS: Cleaning up...', end='')
        os.remove(model_dict['dest'] + '/test_dump.t7')
        os.remove(dataset_dict['dest'] + '/Out_13_1337.png')
        print(done)
        print('[I] STATUS: All pre-flight tests passed! All essential'
              ' sub-systems are green!')


def tdu_cls():

    CLASSES = ['wall', 'floor', 'ceiling', 'bed', 'window', 'cabinet', 'door',
               'table (excluding coffee-table, side-table, sofa-table, desk), end-table, occasional-table', 'plant',
               'curtain, drape, valance', 'chair (excluding accent-chair, armchair, recliner)',
               'painting, poster, photo', 'sofa, couch', 'mirror', 'rug, carpet, mat',
               'accent-chair, armchair, recliner', 'desk', 'wardrobe, hall-tree, closet', 'table-lamp, floor-lamp',
               'bathtub', 'throw-pillow, decorative-pillow, pillow, cuschion, floor-pillow', 'boxes, basket',
               'dresser, chest', 'counter, countertop, kitchen-island', 'sink', 'fireplace', 'fridge, refrigerator',
               'stair', 'bookshelf, decorative-ledge, bookcase', 'window-blind, window-shutter',
               'coffe-table, side-table, sofa-table', 'toilet', 'book', 'kitchen-stove', 'computer, laptop, notebook',
               'swivel-chair', 'towel', 'overhead-lighting, chandelier, pendant, pendent',
               'tv, tv-screen, screen, monitor, television', 'cloth', 'fence, bannister, balauster, handrail',
               'ottoman, footstool', 'bottle', 'washer, washing-machine, dryer, drying-machine', 'game, puzzle, toy',
               'bag', 'oven, furnace', 'microwave, microwave-oven', 'flowerpot', 'bicycle', 'dishwasher',
               'blanket, throw, sofa-blanket', 'kitchen-air-extractor, hood, exhaust-hood', 'sconce, wall-sconce',
               'bin', 'fan', 'shower', 'radiator', 'wall-clock, father-clock', 'window-frame', 'door-frame',
               'decor-accent, table-clock, candle, candleholder, lantern, bookend', 'wall-decoration, art',
               'curtain-rod', 'sound-system, audio-system, speaker, loud-speaker, sound-box, sounding-box, stereo',
               'piano', 'guitar', 'wall-switch', 'room-divider', 'telephone', 'fireplace-screen', 'dog-bed, cat-bed',
               'kitchen-utensil', 'crockery, dish', 'cutting-board', 'pan, kitchen-pot',
               'magazine-rack, magazine-basket', 'coat-rack', 'fireplace-tool', 'sport-machine', 'tripod', 'printer',
               'wire', 'keyboard', 'mouse', 'pad', 'bed-frame', 'balcony', 'stuff', 'board', 'toilet-paper', 'heater',
               'receiver', 'remote', 'hanger', 'soap-dispenser', 'plug', 'flush-button', 'alarm', 'shoe-rack', 'shoe',
               'hair-dryer', 'temperature-controller', 'pipe', 'charger', 'ironing-table', 'shower-head', 'cage', 'hat',
               'vacuum-cleaner', 'tent', 'drum, drum-stick', 'toilet-brush', 'baggage, luggage, suitcase', 'door-glass',
               'tv-unit', 'water-pump', 'stand', 'storage', 'unknown']

    return CLASSES

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

def batched_normalized_rgb_tester(img_batch):
    b, c, w, h = img_batch.shape
    normalized_img_batch = torch.zeros(b, c, w, h)

    for batch_targ in range(b):
        R = img_batch[batch_targ, 0, :, :]
        G = img_batch[batch_targ, 1, :, :]
        B = img_batch[batch_targ, 2, :, :]

        dino = R + G + B

        norm_R = torch.nan_to_num(R / dino, nan=0.0)
        norm_G = torch.nan_to_num(G / dino, nan=0.0)
        norm_B = torch.nan_to_num(B / dino, nan=0.0)

        normalized_img_batch[batch_targ, 0, :, :] = norm_R
        normalized_img_batch[batch_targ, 1, :, :] = norm_G
        normalized_img_batch[batch_targ, 2, :, :] = norm_B

    return normalized_img_batch


