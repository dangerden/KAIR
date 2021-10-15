import os.path
import math
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

def dewatermark(sample_wm, pv_logo, alpha=0.7):
	s_height, s_width, s_channels = sample_wm.shape

	logo_scale = 0.6
	l_h, l_w = pv_logo.shape[:2]

	logo_bar_w = s_width
	logo_bar_h = s_width / 6

	wm_x = logo_bar_h / 2
	wm_y = s_height - logo_bar_h / 2
	wm_size = logo_bar_h * logo_scale

	wm_size = np.round(wm_size)

	scale = wm_size / logo_bar_w
	wm_origin_x = wm_x - wm_size / 2
	wm_origin_y = wm_y - wm_size / 2

	wm_origin_x = np.round(wm_origin_x)
	wm_origin_y = np.round(wm_origin_y)

	M = np.float32([[wm_size / l_w, 0, wm_origin_x],[0, wm_size / l_h, wm_origin_y]])
	overlay = cv2.warpAffine(pv_logo, M, (s_width, s_height), cv2.INTER_AREA) # interpolation doesn't seem to affect anything

	dwm = np.copy(sample_wm)
	mask = np.zeros((s_height, s_width), np.uint8)
	wm_size0 = int(np.ceil(wm_size)) + 1
	wm_x0 = int(np.floor(wm_origin_x))
	wm_y0 = int(np.floor(wm_origin_y))
	for ix0, iy0 in np.ndindex((wm_size0, wm_size0)):
		ix = wm_x0 + ix0; iy = wm_y0 + iy0
		src_pix = dwm[iy, ix]
		logo_pix = overlay[iy, ix]
		mask[iy, ix] = 1 - (np.array_equal(logo_pix, overlay[iy + 1, ix]) and np.array_equal(logo_pix, overlay[iy - 1, ix]) and np.array_equal(logo_pix, overlay[iy, ix + 1]) and np.array_equal(logo_pix, overlay[iy, ix - 1]))
		dst_pix = (src_pix - logo_pix[:3] * alpha * logo_pix[3]/255) / (1 - alpha * logo_pix[3]/255)
		dwm[iy, ix] =  np.clip(dst_pix, 0, 255)

	telea = cv2.inpaint(dwm, mask, 3, cv2.INPAINT_TELEA)
	return telea

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
            img_H = cv2.imread(H_path)
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
              wm_size = math.ceil(logo_bar_h * 0.6)
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
