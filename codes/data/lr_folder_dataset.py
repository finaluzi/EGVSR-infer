import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files
from pathlib import Path

import time


class LRFolderDataset(BaseDataset):
    def __init__(self, data_opt, opt_out, **kwargs):
        """ Folder dataset with paired data
            support both BI & BD degradation
        """
        super(LRFolderDataset, self).__init__(data_opt, **kwargs)

        lr_keys = sorted(os.listdir(self.lr_seq_dir))
        self.keys = sorted(list(set(lr_keys)))
        # filter keys
        if self.filter_file:
            with open(self.filter_file, 'r') as f:
                sel_keys = {line.strip() for line in f}
                self.keys = sorted(list(sel_keys & set(self.keys)))
        self.batch_num = 0
        self.cache_num = opt_out['test']['cache_length']
        self.file_list_dict = {}
        for k in self.keys:
            self.file_list_dict[k] = retrieve_files(
                osp.join(self.lr_seq_dir, k))
        self.pad_num = opt_out['test']['num_pad_front']

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        start_idx = self.batch_num*self.cache_num
        end_idx = (self.batch_num+1)*self.cache_num
        frames_in_dir = len(self.file_list_dict[key])
        if frames_in_dir > end_idx and frames_in_dir-end_idx <= self.pad_num:
            end_idx = frames_in_dir
            print('[B] Combine batch', self.batch_num, '&',
                  self.batch_num+1)
        print('[B] batch:', self.batch_num,
              'start:', start_idx, 'end:', end_idx, 'all:', frames_in_dir)
        # load lr frames
        lr_seq = []
        file_list = self.file_list_dict[key][start_idx:end_idx]

        file_names = list(map(lambda x: Path(x).name, file_list))
        for frm_path in file_list:
            frm = cv2.imread(frm_path)
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq)  # thwc|rgb|uint8
        lr_seq = lr_seq[..., ::-1]
        lr_tsr = torch.from_numpy(
            np.ascontiguousarray(lr_seq)).to(torch.float32)/255.0
        return {
            'lr': lr_tsr,
            'seq_idx': key,
            'frm_idx': file_names
        }

    def is_end(self):
        start_idx = self.batch_num*self.cache_num
        for _, v in self.file_list_dict.items():
            if start_idx < len(v):
                return False
        return True
