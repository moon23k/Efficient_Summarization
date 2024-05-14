import json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']



class Collator(object):
    def __init__(self, config):
        self.pad_id = config.pad_id
        self.max_len = config.full_len


    def pad_batch(self, batch):
        x_batch, y_batch = zip(*batch)

        pad_sequence = lambda seq, max_len: [x + [self.pad_id] * (max_len - len(x)) for x in seq]

        y_max_len = max(len(seq) for seq in y_batch)

        x_batch = pad_sequence(x_batch, self.max_len)
        y_batch = pad_sequence(y_batch, y_max_len)
        
        return torch.LongTensor(x_batch), torch.LongTensor(y_batch)


    def __call__(self, batch):
        x_batch, y_batch = self.pad_batch(batch)
        
        return {'x': x_batch,
                'y': y_batch}



def load_dataloader(config, split):
    return DataLoader(
        Dataset(split),
        batch_size=config.batch_size, 
        shuffle=True if split == 'train' else False,
        collate_fn=Collator(config),
        num_workers=2
    )
    