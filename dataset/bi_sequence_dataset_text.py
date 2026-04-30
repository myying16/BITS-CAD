from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import numpy as np
import csv
from config.macro import *

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    dataset = BiSequenceDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader

class BiSequenceDataset(Dataset):
    def __init__(self, phase, config):
        super(BiSequenceDataset, self).__init__()
        self.svg_vec = os.path.join(config.data_root, "svg_vec")
        self.cad_vec = os.path.join(config.data_root, "cad_vec")
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            all_data = json.load(fp)[phase]
        self.svg_max_total_len = SVG_MAX_TOTAL_LEN #100
        self.cad_max_total_len = CAD_MAX_TOTAL_LEN #60

        self.text_csv_path = os.path.join(config.data_root, "text2cad.csv")
        self.text_dict = {}  
        with open(self.text_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_id = row["uid"]
                text = row["expert"]
                self.text_dict[data_id] = text

        #Only preserve CAD with text.
        self.all_data = [
            data_id for data_id in all_data
            if data_id in self.text_dict]
        print(f"[{phase}] total CAD: {len(all_data)}, with text: {len(self.all_data)}")


    def __len__(self):
        return len(self.all_data)
    
    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def get_svg_data(self, data_id):
        npy_path = os.path.join(self.svg_vec, data_id + ".npy")
        data = np.load(npy_path)

        command_vec = data[300:, 1]
        args_vec = data[300:, 2:]
        command_vec = torch.tensor(command_vec, dtype=torch.long)
        args_vec = torch.tensor(args_vec, dtype=torch.long)
        return {"command": command_vec, "args": args_vec}


    def get_cad_data(self, data_id):
        h5_path = os.path.join(self.cad_vec, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len=60, 1 + N_ARGS=17)

        pad_len = self.cad_max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, CAD_EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]  # (60,)
        args = cad_vec[:, 1:]  # (60, 16)
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args}


    def __getitem__(self, index):
        data_id = self.all_data[index]
        cad_data = self.get_cad_data(data_id)
        svg_data = self.get_svg_data(data_id)
        text = self.text_dict.get(data_id)
        return {"svg": svg_data, "cad": cad_data, "text":text, "id": data_id}
