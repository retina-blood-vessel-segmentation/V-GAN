from utils import all_files_under, chopup, integrate
import numpy as np
from pathlib import Path
from itertools import product
import multiprocessing as mp
import os



def load(code):
    global countermap
    global outdir
    global p
    global buffer
    global indexCounter
    global index
    global backIndex


    if code in index:
        i = index[code]
    else:
        i = indexCounter
        if i in backIndex:
            oldcode = backIndex[i]
            index.pop(oldcode, None)
            backIndex.pop(i, None)
        index[code] = i
        backIndex[i] = code
        buffer[i, :, :, :3] =  np.load(str(p / "images" / f'{code}.npy'))
        buffer[i, :, :, 3] =  np.load(str(p / "masks" / f'{code}.npy'))
        buffer[i, :, :, 4] =  np.load(str(p / "labels" / f'{code}.npy'))
        indexCounter = indexCounter + 1 
        if indexCounter >= 512:
            indexCounter = 0       
    return buffer[i, :, :, :3],buffer[i, :, :, 3],buffer[i, :, :, :4]
    

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def prepare(chunk):
    global countermap
    global outdir
    global p
    global buffer
    global indexCounter
    global index
    global backIndex

    pid = str(mp.current_process().pid)
    if pid not in countermap:
        countermap[pid] = 0
    counter = countermap[pid]
    name = str(outdir / f'{pid}_{counter}.npy')
    rez = np.zeros((nc*nc, s, s, 5))
    countermap[pid] = countermap[pid] + 1
    k = 0
    for i in range(nc):
        for j in range(nc):
            b = chunk[k]
            img = np.load(str(p / "images" / f'{b[0]}.npy'))
            msk = np.load(str(p / "masks" / f'{b[0]}.npy'))
            gnd = np.load(str(p / "labels" / f'{b[0]}.npy'))
            #img, msk, gnd = load(b[0])
            rez[k, :, :, :3] = img[b[1]*s:(b[1] + 1)*s, b[2]*s:(b[2] + 1)*s, ...]
            rez[k, :, :, 3] = msk[b[1]*s:(b[1] + 1)*s, b[2]*s:(b[2] + 1)*s, ...]
            rez[k, :, :, 4] = gnd[b[1]*s:(b[1] + 1)*s, b[2]*s:(b[2] + 1)*s, ...]
            k = k + 1
    np.save(name, rez)
    print(f'Processed {name}')
    countermap[pid] = countermap[pid] + 1
            

if __name__ == '__main__':
    global countermap
    global outdir
    global p
    global buffer
    global indexCounter
    global index
    global backIndex

    countermap={}
    outdir = Path("/home/veljko/trainset")
    p = Path("/home/veljko/aug")
    #buffer=np.zeros((512,1024,1024,5))
    indexCounter = 0
    index = {}
    backIndex = {}

    s = 128 ##chunksize
    img_size = (1024,1024)
    nc = img_size[0] // s

    imgfs_o = all_files_under(str(p / "images"), None, True, True)
    n=len(imgfs_o)
    fs = [Path(img).stem for img in imgfs_o]
    data = []
    for f in fs:
        for i in range(8):
            for j in range(8):
                data.append((f, i, j))  
    ## Data - A combination of every file-code & all possible combos of 0..nc
    ## A random 64-element (nc-squared) chunk thereof is one training chunk.
    np.random.seed(0xae08b357) ##Guaranteed random, but repeatable for future use. 
    indices = np.random.choice(n * nc * nc, n * nc * nc, replace=False)
    data2 = [data[i] for i in indices]
    chunks = list(chunkify(data2, nc * nc))
    with mp.Pool(1) as pl:
        pl.map(prepare, chunks, chunksize=1)
    
    pl.join()

    
