import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


ALL_KEYS = [
    'Date', 
    'EUA_1', 'EUA_2', 'EUA_3', 'EUA_4', 
    'Brent_1', 'Brent_2', 'Brent_3', 'Brent_4', 
    'Coal_1', 'Coal_2', 'Coal_3', 'Coal_4',
    'Diesel_1', 'Diesel_2', 'Diesel_3', 'Diesel_4', 
    'Gasoline_1', 'Gasoline_2', 'Gasoline_3', 'Gasoline_4', 
    'Industrial_1', 'Industrial_2', 'Industrial_3', 'Industrial_4', 
    'CAC40_1', 'CAC40_2', 'CAC40_3', 'CAC40_4', 
    'DAX30_1', 'DAX30_2', 'DAX30_3', 'DAX30_4',
    'FTSE100_1', 'FTSE100_2', 'FTSE100_3', 'FTSE100_4', 
    'STOXX50_1', 'STOXX50_2', 'STOXX50_3', 'STOXX50_4', 
    'STOXX600_1', 'STOXX600_2', 'STOXX600_3', 'STOXX600_4', 
    'gas_spot', 
    'Temperature_Mean', 'Temperature_High', 'Temperature_Low', 
    'Extreme_Temperature_Mark',
    'COVID19_Increase', 'COVID19_Total'
]

XS_KEYS = [
    'Brent_1', 'Brent_2', 'Brent_3', 'Brent_4', 
    'Coal_1', 'Coal_2', 'Coal_3', 'Coal_4',
    'Diesel_1', 'Diesel_2', 'Diesel_3', 'Diesel_4', 
    'Gasoline_1', 'Gasoline_2', 'Gasoline_3', 'Gasoline_4', 
    'Industrial_1', 'Industrial_2', 'Industrial_3', 'Industrial_4', 
    'CAC40_1', 'CAC40_2', 'CAC40_3', 'CAC40_4', 
    'DAX30_1', 'DAX30_2', 'DAX30_3', 'DAX30_4',
    'FTSE100_1', 'FTSE100_2', 'FTSE100_3', 'FTSE100_4', 
    'STOXX50_1', 'STOXX50_2', 'STOXX50_3', 'STOXX50_4', 
    'STOXX600_1', 'STOXX600_2', 'STOXX600_3', 'STOXX600_4', 
    'gas_spot', 
    'Temperature_Mean', 'Temperature_High', 'Temperature_Low', 
    'Extreme_Temperature_Mark',
    'COVID19_Increase', 'COVID19_Total'
]

YS_KEYS = [
    'EUA_1', 'EUA_2', 'EUA_3', 'EUA_4'
]


def load_csv(file_name: str):
    data = pd.read_csv(file_name)

    return data


class MyDataset(Dataset):

    def __init__(self, file_path: str, seq_len=4, istrain=True):
        self.file_path = file_path
        self.seq_len = seq_len
        self.istrain = istrain
        self.data = self._load_data()
    
    def _load_data(self):
        data = load_csv(self.file_path)
        return data

    def __getitem__(self, index):
        # generate indexes
        inds = [_ for _ in range(index, index+self.seq_len)]
        inds = [min(_, len(self.data)) for _ in inds]

        dataset_dict = {}
        data_item = [self.data.iloc[idx] for idx in inds]
        labels = np.asarray([
            [_[key] for key in YS_KEYS]
            for _ in data_item
        ])
        dataset_dict["labels"] = torch.from_numpy(labels.astype(np.float32))
        inputs = np.asarray(
            [
                [_[key] for key in XS_KEYS]
                for _ in data_item
            ]
        )
        dataset_dict["inputs"] = torch.from_numpy(inputs.astype(np.float32))

        return dataset_dict
    
    def __len__(self):
        return len(self.data) - self.seq_len + 1


def build_train_loader(file_path, istrain=True):
    dataset = MyDataset(file_path=file_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # for batched_inputs in iter(data_loader):
    #     import ipdb; ipdb.set_trace()
    return data_loader
    

if __name__ == "__main__":
    build_train_loader(file_path="data/constrained.csv")
