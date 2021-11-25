import os.path
import math
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

class DatasetPeakSR(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for superresolution on PeakVisor dataset.
    # 
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetPeakSR, self).__init__()
        print('Dataset: PeakVisor structured superresolution dataset.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 512
        
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        if self.opt['phase'] == 'train':
            self.paths_H = np.repeat(self.paths_H, 10)
            np.random.shuffle(self.paths_H)
        self.wmark = cv2.imread(opt['watermark'], cv2.IMREAD_UNCHANGED)
        
    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """

            # --------------------------------
            # train on smaller images (should we?)
            # --------------------------------
            try:
                img_H = cv2.imread(H_path)
            except:
                print(f"Failed to decode {H_path}... Randomly trying another file.")
                return self.__getitem__(np.random.randint(0, 128))

            H, W, _ = img_H.shape
            img_H = cv2.resize(img_H, (1024, round(1024*H/W)) , interpolation=cv2.INTER_LANCZOS4)
            H, W, _ = img_H.shape

            # --------------------------------
            # algorithmic removal
            # --------------------------------
            if H_path.find('sharing_rnd') != -1:
              img_L = np.copy(img_H)
              img_H = dewatermark(img_L, self.wmark)
              #
              img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
              img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)

            else:
            # --------------------------------
            # randomly watermark
            # --------------------------------
              WM_H, WM_W, _ = self.wmark.shape
              logo_bar_w, logo_bar_h = W, W/6
              wm_size = math.ceil(logo_bar_h * 0.6) + random.randint(-20, 20)
              wm_x, wm_y = random.randint(0, max(0, W - wm_size)), random.randint(0, max(0, H - wm_size))
              
              logo_resized = cv2.resize(self.wmark, (wm_size, wm_size), interpolation=cv2.INTER_LANCZOS4)
              logo_alpha_slice = 0.7 * logo_resized[:,:,3:] / 255.
              premul_logo = logo_resized[:,:,:3] * logo_alpha_slice

              alpha_pad = np.zeros_like(img_H, dtype=float)
              alpha_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = logo_alpha_slice
              overlay_pad = np.zeros_like(img_H, dtype=float)
              overlay_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = premul_logo

              img_L = img_H.astype(float)*(1-alpha_pad)+overlay_pad

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            #mode = random.randint(0, 7)
            #patch_H = util.augment_img(patch_H, mode=mode)
            
            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_L = cv2.imread(L_path)
            H, W, _ = img_L.shape
            img_L = cv2.resize(img_L, (1024, round(1024*H/W)) , interpolation=cv2.INTER_LANCZOS4)
            img_H = dewatermark(img_L, self.wmark)
            #
            img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
            # --------------------------------
            # PeakVisor specific
            # --------------------------------
            #WM_H, WM_W, _ = self.wmark.shape
            #logo_bar_w, logo_bar_h = W, W/6
            #wm_size = math.ceil(logo_bar_h * 0.6)
            #wm_x, wm_y = round(logo_bar_h/2), round(H - logo_bar_h/2 - wm_size/2)
            #logo_resized = cv2.resize(self.wmark, (wm_size, wm_size), interpolation=cv2.INTER_LANCZOS4)
            #logo_alpha_slice = 0.7 * logo_resized[:,:,3:] / 255.
            #premul_logo = logo_resized[:,:,:3] * logo_alpha_slice / 255.

            #alpha_pad = np.zeros_like(img_H)
            #alpha_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = logo_alpha_slice
            #overlay_pad = np.zeros_like(img_H)
            #overlay_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = premul_logo

            #img_L = img_H*(1-alpha_pad)+overlay_pad

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_H = util.uint2tensor3(img_H)
            img_L = util.uint2tensor3(img_L)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
