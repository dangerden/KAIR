import sys
import os.path
import logging
import time
import re
import math
from collections import OrderedDict
import torch

from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

from utils import utils_logger
from utils import utils_image as util

'''
Process PeakVisor map tiles into manageable patches
esdsat - ultra high resolution (~1m per pixel)
hdsat - high resolution (~10m per pixel)
hddem - Digital Elevation Model (~30m grid)
'''

def main():

    utils_logger.logger_info('PeakVisor SR SAT', log_path='peak_sr.log')
    logger = logging.getLogger('PeakVisor SR SAT')

    SRC_ROOT = "/Users/denisbulichenko/Downloads/demo64"
    DST_ROOT = "/Users/denisbulichenko/Downloads/pksr_dataset"
    HQ_PATCH_SZ = 512
    LQ_PATCH_SZ = HQ_PATCH_SZ // 4

    Image.MAX_IMAGE_PIXELS = None
    
    util.mkdir(DST_ROOT)
    
    # 1. Iterate map tiles in a primary directory, e.g. esdsat
    paths = util.get_image_paths(f"{SRC_ROOT}/esdsat")
    logger.info(f"{len(paths)} map tiles found in {SRC_ROOT}/esdsat")

    tile_re = re.compile('.*\/tESDSAT_c([A-Z0-9]*)_.*')

    for i in tqdm(range(len(paths))):
        img = paths[i]
        #logger.info(f"Tile [{i}] => {img}")
        
        # 2. Parse map tile index - regexp
        tile_c = tile_re.match(img).group(1)

        esd_file = f"{SRC_ROOT}/esdsat/tESDSAT_c{tile_c}_v0_fJPG.tif"
        hd_file = f"{SRC_ROOT}/hdsat/tHDSAT_c{tile_c}_v3_fJPG.jpg"
        dem_file = f"{SRC_ROOT}/hddem/tHDDEM_c{tile_c}_vis.tif"

        esd_im = np.asarray(Image.open(esd_file))
        hd_im = np.asarray(Image.open(hd_file))
        dem_im = np.asarray(Image.open(dem_file))[:,:,[0]]

        e_h, e_w, e_c = esd_im.shape
        l_h, l_w, l_c = hd_im.shape
        d_h, d_w, d_c = dem_im.shape
        #logger.info(f"Original data ESD=({e_w}x{e_h}) HD=({l_w}x{l_h}) DEM=({dem_im.shape})")
        #logger.info(f"Target sq patches ESD={HQ_PATCH_SZ}px HD={math.ceil(l_h*HQ_PATCH_SZ/e_h)}x{math.ceil(l_w*HQ_PATCH_SZ/e_w)}px DEM={math.ceil(d_w*HQ_PATCH_SZ/e_w)}x{math.ceil(d_h*HQ_PATCH_SZ/e_h)}px")

        # 3. Split into patches
        hq_patch = util.patches_from_image(esd_im, p_size=HQ_PATCH_SZ, p_overlap=64)
        #logger.info(f"HQ patches len={len(hq_patch)}, shape=[{hq_patch[0].shape}]")
        # LQ patches will be 4 times smaller
        lq_res_im = cv2.resize(hd_im, dsize=(math.ceil(e_w/4), math.ceil(e_h/4)), interpolation=cv2.INTER_CUBIC)
        lq_patch = util.patches_from_image(lq_res_im, p_size=HQ_PATCH_SZ//4, p_overlap=64//4)
        #logger.info(f"LQ patches len={len(lq_patch)}, shape=[{lq_patch[0].shape}]")
        # Bicubic LQ
        blq_res_im = cv2.resize(esd_im, dsize=(math.ceil(e_w/4), math.ceil(e_h/4)), interpolation=cv2.INTER_CUBIC)
        blq_patch = util.patches_from_image(blq_res_im, p_size=HQ_PATCH_SZ//4, p_overlap=64//4)
        #logger.info(f"Bicubic LQ patches len={len(blq_patch)}, shape=[{blq_patch[0].shape}]")
        # DEM patches
        dem_res_im = cv2.resize(dem_im, dsize=(math.ceil(e_w/4), math.ceil(e_h/4)), interpolation=cv2.INTER_CUBIC)[..., np.newaxis]
        dem_patch = util.patches_from_image(dem_res_im, p_size=HQ_PATCH_SZ//4, p_overlap=64//4)
        #logger.info(f"DEM patches len={len(dem_patch)}, shape=[{dem_patch[0].shape}]")

        assert len(hq_patch)==len(lq_patch) , "Resulted in different amount of patches for HQ and LQ"

        # 4. Save patches
        util.mkdir(f"{DST_ROOT}/{tile_c}")
        for i in range(len(hq_patch)):
            cv2.imwrite(f"{DST_ROOT}/{tile_c}/{tile_c}-p-{i:05d}-hq.png", hq_patch[i])
            cv2.imwrite(f"{DST_ROOT}/{tile_c}/{tile_c}-p-{i:05d}-lq-ds.png", blq_patch[i])
            cv2.imwrite(f"{DST_ROOT}/{tile_c}/{tile_c}-p-{i:05d}-lq.png", lq_patch[i])
            cv2.imwrite(f"{DST_ROOT}/{tile_c}/{tile_c}-p-{i:05d}-dem.png", dem_patch[i])




if __name__ == '__main__':

    main()
