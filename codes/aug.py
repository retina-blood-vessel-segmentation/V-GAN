import os
import sys

from PIL import Image, ImageEnhance
from scipy.ndimage import rotate

import numpy as np
import utils
import multiprocessing as mp
from pathlib import Path

outpath = Path("/home/veljko/aug")
smallify=False
debug = False

def save(where, i, img):
    p = outpath / where / f'{i:06d}.npy'
    np.save(p, img)

def saveLinked(where, i, j):
    orig = (outpath /where / f'{i:06d}.npy').resolve()
    link = outpath /where / f'{j:06d}.npy'
    os.symlink(orig, link)

def normalize(img):
    mean = np.mean(img[...][img[...,0] > 40.0], axis = 0)
    std = np.std(img[...][img[...,0] > 40.0], axis = 0)
    img = (img[...] - mean) / std
    return img

def makeSmall(img):
    if not smallify:
        return img
    newx = int(img.shape[0] * 0.48)
    newy = int(img.shape[1] * 0.48)
    return cv2.resize(img, (newx,newy), interpolation = cv2.INTER_AREA)


def augment(imgfs, lblfs, mskfs):
    img_size = (1024, 1024) if not smallify else (480,480)
    i, imgf = imgfs
    _, lblf = lblfs
    _, mskf = mskfs
    img = utils.imagefiles2arrs([imgf])
    lbl = utils.imagefiles2arrs([lblf])
    msk = utils.imagefiles2arrs([mskf])
    img = makeSmall(img)
    lbl = makeSmall(lbl)
    msk = makeSmall(msk)
    img = utils.pad_imgs(img, img_size)[0]
    lbl = utils.pad_imgs(lbl, img_size)[0]
    msk = utils.pad_imgs(msk, img_size)[0]
    #lbl = lbl / 255
    #msk = msk / 255
    j = i * 10000
    masknum = j
    save("masks", j, msk)
    for angle in range(0,360, 3):
        img2 = np.copy(img)
        lbl2 = np.copy(lbl)
        #msk2 = np.copy(msk)
        img3 = img[:,::-1, :]
        lbl3 = lbl[:, ::-1]
        #msk3 = msk[:, ::-1]
        img4 = np.copy(img3)
        lbl4 = np.copy(lbl3)
        #msk4 = np.copy(msk3)
        if angle > 0:
            img2 = utils.random_perturbation(rotate(img, angle, axes=(1,0), reshape = False))
            img4 = utils.random_perturbation(rotate(img3, angle, axes=(1,0), reshape = False))
            lbl2 = rotate(lbl, angle, axes=(1,0), reshape = False)
            lbl4 = rotate(lbl3, angle, axes=(1,0), reshape = False)
            lbl2[lbl2 <= 0.5] = 0.0
            lbl2[lbl2 > 0.5] = 1.0
            lbl4[lbl4 <= 0.5] = 0.0
            lbl4[lbl4 > 0.5] = 1.0
            #msk2 = rotate(msk, angle, axes=(1,0), reshape = False)
            #msk4 = rotate(msk3, angle, axes=(1,0), reshape = False)
        img2 = normalize(img2)
        img4 = normalize(img4)
        save("images", j, img2)
        save("labels", j, lbl2)
        if j != masknum:
            saveLinked("masks", masknum, j)
        j = j + 1
        save("images", j, img4)
        save("labels", j, lbl4)
        if j != masknum:
            saveLinked("masks", masknum, j)
        if debug:
            im = Image.fromarray((img2*255).astype(np.uint8))
            im.save(str(outpath / f'img2_{j:06d}.png'))
            im = Image.fromarray((img4*255).astype(np.uint8))
            im.save(str(outpath / f'img4_{j:06d}.png'))
            im = Image.fromarray((lbl2*255).astype(np.uint8))
            im.save(str(outpath / f'lbl2_{j:06d}.png'))
            im = Image.fromarray((lbl4*255).astype(np.uint8))
            im.save(str(outpath / f'lbl4_{j:06d}.png'))
        j = j + 1
    print(f'File {i} processed.')

if __name__ == '__main__':

    imgfs = list(enumerate(utils.all_files_under("/home/shared/retina/IterNet/data/CHASE/training/images", append_path=True, sort=True)))
    lblfs = list(enumerate(utils.all_files_under("/home/shared/retina/IterNet/data/CHASE/training/1st_manual", append_path=True, sort=True)))
    mskfs = list(enumerate(utils.all_files_under("/home/shared/retina/IterNet/data/CHASE/training/mask", append_path=True, sort=True)))

    assert len(imgfs) == len(lblfs) and len(lblfs) == len(mskfs), "Problem?"

    d = list(zip(imgfs, lblfs, mskfs))
    with mp.Pool(8) as p:
        chunk = len(imgfs) // 8
        p.starmap(augment, d)

    p.join()