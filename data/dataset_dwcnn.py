import os.path
import math
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDwCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for dewatermarking on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DwCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDwCNN, self).__init__()
        print('Dataset: Dewatermarking of PeakVisor logo. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

        self.wmark = util.imread_uint("G:/My Drive/demark/sh-logo.png", 4)
        
    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        H, W, _ = img_H.shape
        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            
            # -
            # randomly watermark
            # ---
            WM_H, WM_W, _ = self.wmark.shape
            wm_x, wm_y = random.randint(0, max(0, W - WM_W)), random.randint(0, max(0, H - WM_H))
            logo_bar_w, logo_bar_h = W, H/6
            wm_size = math.ceil(logo_bar_h * 0.6)
            logo_resized = util.imresize(self.wmark, WM_H/wm_size)
            logo_alpha_slice = 0.7 * logo_resized[:,:,3:]
            premul_logo = logo_resized[:,:,:3] * logo_alpha_slice

            alpha_pad = torch.zeros_like(img_H)
            alpha_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = logo_alpha_slice
            overlay_pad = torch.zeros_like(img_H)
            overlay_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = premul_logo
            img_L = img_H*(1-alpha_pad)+overlay_pad

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
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            
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
            img_H = util.uint2single(img_H)

            # -
            # randomly watermark
            # ---
            WM_H, WM_W, _ = self.wmark.shape
            logo_bar_w, logo_bar_h = W, H/6
            wm_x, wm_y = logo_bar_h/2, WM_H - logo_bar_h/2
            wm_size = math.ceil(logo_bar_h * 0.6)
            logo_resized = util.imresize(self.wmark, WM_H/wm_size)
            logo_alpha_slice = 0.7 * logo_resized[:,:,3:]
            premul_logo = logo_resized[:,:,:3] * logo_alpha_slice

            alpha_pad = torch.zeros_like(img_H)
            alpha_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = logo_alpha_slice
            overlay_pad = torch.zeros_like(img_H)
            overlay_pad[wm_y:wm_y+wm_size, wm_x:wm_x+wm_size, :] = premul_logo
            img_L = img_H*(1-alpha_pad)+overlay_pad

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)
            print("SHOULDN'T BE HERE!")

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
