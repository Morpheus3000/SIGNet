from torch.utils.data import Dataset
import numpy as np
import imageio
import cv2


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


def colourize_semantics(seg_img, palette):
    colour_seg = np.zeros((seg_img.shape[0], seg_img.shape[1], 3),
                          dtype=np.uint8)
    for label, colour in enumerate(palette):
        colour_seg[seg_img == label, :] = colour
    return colour_seg


def average_pixel(img, raw_seg, palette):
    avg_pix = np.zeros_like(img)
    for label, colour in enumerate(palette):
        sel_pixels = raw_seg == label
        if np.sum(sel_pixels) > 0:
            avg_pix[sel_pixels, :] = np.mean(img[sel_pixels, :], 0, keepdims=True)
    return avg_pix


class GeneralDataset(Dataset):

    def __init__(self, data_file, prefix, gray=True):
        self.prefix = prefix
        self.data_paths = self._read_data_file(data_file)
        self.gray = gray
        self.name_fmt = self.prefix + '%s.png'
        self.palette = np.array(tdu_palette())

    def _read_data_file(self, data_file_path):
        filer = open(data_file_path, 'r')
        lines = filer.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        targ = self.data_paths[index]

        # Read rgb
        im = np.asarray(imageio.imread(self.name_fmt % targ))
        rgb = im.astype(np.float32)
        rgb = rgb[:, :, :3]
        rgb[np.isnan(rgb)] = 0
        # rgb = cv2.resize(rgb, (256, 256))
        rgb = rgb / 255

        # Read semantic
        im = np.asarray(imageio.imread(self.prefix + 'seg_' + targ + '.png'))
        raw_seg = im.astype(np.uint8) - 1
        raw_seg[np.isnan(raw_seg)] = 0
        seg_img = colourize_semantics(raw_seg, self.palette).astype(np.float32)
        # raw_seg = cv2.resize(raw_seg, (256, 256))
        seg_img = seg_img / 255
        avg_img = average_pixel(rgb, raw_seg, self.palette)

        seg_img = seg_img.transpose((2, 0, 1))
        avg_img = avg_img.transpose((2, 0, 1))
        rgb = rgb.transpose((2, 0, 1))

        image_dict = {'rgb': rgb,
                      'raw_sem': raw_seg,
                      'inv_img': avg_img,
                      'segs': seg_img}

        return image_dict, self.data_paths[index]


class IIWDataset(Dataset):

    def __init__(self, data_file, prefix, gray=True):
        self.prefix = prefix
        self.data_paths = self._read_data_file(data_file)
        self.gray = gray
        self.name_fmt = self.prefix + '%s.png'
        self.palette = np.array(tdu_palette())

    def _read_data_file(self, data_file_path):
        filer = open(data_file_path, 'r')
        lines = filer.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        file = self.prefix + '/resized/' + self.data_paths[index]

        # Read rgb
        im = np.asarray(imageio.imread(file + '.png'))
        rgb = im.astype(np.float32)
        rgb[np.isnan(rgb)] = 0
        rgb = rgb / 255

        # Read semantic
        im = np.asarray(imageio.imread(self.prefix +
                                       '/IIW_resized_Mask2Former/seg_' +
                                       self.data_paths[index] + '.png'))
        raw_seg = im.astype(np.uint8) - 1
        raw_seg[np.isnan(raw_seg)] = 0
        seg_img = colourize_semantics(raw_seg, self.palette).astype(np.float32)
        # raw_seg = cv2.resize(raw_seg, (256, 256))
        seg_img = seg_img / 255
        # print('rgb: ', rgb.shape)
        # print('seg: ', seg_img.shape)
        # print('raw: ', raw_seg.shape)
        avg_img = average_pixel(rgb, raw_seg, self.palette)
        shd_est = np.nan_to_num(rgb / avg_img)
        shd_est = shd_est / shd_est.max()
        shd_est = cv2.cvtColor(shd_est, cv2.COLOR_RGB2GRAY)

        seg_img = seg_img.transpose((2, 0, 1))
        avg_img = avg_img.transpose((2, 0, 1))
        rgb = rgb.transpose((2, 0, 1))
        shd_est = shd_est[np.newaxis, :, :]

        image_dict = {'rgb': rgb,
                      'raw_sem': raw_seg,
                      'inv_img': avg_img,
                      'shd_est': shd_est,
                      'segs': seg_img}

        return image_dict, self.data_paths[index]


if __name__ == '__main__':
    # Sample dataloader example. The sample below loads one random image from
    # the IIW dataset and saves the different types returned by the dataset.

    from torch.utils.data import DataLoader
    import time

    # Point to the data root.
    data_root = '/var/scratch/pdas/Datasets/IIW/'
    judgement_root = data_root + 'judgementDumps/'
    train_list = data_root + 'train_files.txt'

    print('[+] Init dataloader')
    testSet = IIWDataset(train_list, data_root)
    print('[+] Create workers')
    loader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print('[*] Dataset size: ', len(loader))
    enu = enumerate(loader)
    for i in range(1):
        a = time.time()
        i, (images, filer) = next(enu)
        b = time.time()
        inv = images['inv_img'][0, :, :, :].cpu().numpy()
        sem = images['segs'][0, :, :, :].cpu().numpy()
        rgb = images['rgb'][0, :, :, :].cpu().numpy()
        shd_est = images['shd_est'][0, :, :, :].cpu().numpy()
        inv = inv.transpose((1, 2, 0)) * 255
        sem = sem.transpose((1, 2, 0)) * 255
        rgb = rgb.transpose((1, 2, 0)) * 255
        shd_est = shd_est.transpose((1, 2, 0)) * 255
        imageio.imwrite('tmp_img.png', rgb.astype(np.uint8))
        imageio.imwrite('tmp_inv.png', inv.astype(np.uint8))
        imageio.imwrite('tmp_sem.png', sem.astype(np.uint8))
        imageio.imwrite('tmp_shd_est.png', shd_est.astype(np.uint8))
