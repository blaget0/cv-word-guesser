import numpy as np
import os
import torch
import torchvision
import shutil
from tqdm import tqdm
from pathlib import Path

limit = 10000

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, pre=False):
        if pre == True:
            self.data_list = np.load("full_data.npy")
            self.targets = np.load("full_targets.npy")
        else:
            self.data_numpy_list = [x for x in Path('.').glob(os.path.join(root_path, '*.npy'))]
            self.data_list = np.empty(shape=[1,1,784])
            self.targets = np.array([])
            id = 0
            for ind in tqdm(range(len(self.data_numpy_list))):
                data_slice_file_name = self.data_numpy_list[ind]
                data_i = (np.load(data_slice_file_name))[:limit]
                self.data_list = np.append(self.data_list, data_i[:,np.newaxis, :], axis=0)
                self.targets = np.concatenate([self.targets, np.full(data_i.shape[0], id)])
                id += 1
            self.data_list = ((self.data_list).reshape((-1,28,28)))[:, np.newaxis]

    def __getitem__(self, index):
        self.data = np.asarray(self.data_list[index])
        if index > ((self.targets).shape[0] - 1):
            index = index % ((self.targets).shape[0] - 1)
        return (torch.from_numpy(self.data).float(), self.targets[index])

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':  

    data_root = '../cv-word-guesser/data/numpy_bitmap'
    file_names = [name for name in os.listdir(data_root)]
    labels = [name.replace('.npy', '').replace('full_numpy_bitmap_', '') for name in os.listdir(data_root)]
    print(labels)
    '''
    for i, label in enumerate(labels):
        dest_dir = os.path.join(data_root, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(os.path.join(data_root, file_names[i]), os.path.join(dest_dir, label + '.npy'))
    '''