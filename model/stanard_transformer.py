import torch
import torch.nn as nn
from collections import namedtuple
from .components import (
    clones, 
    Embeddings, 
    SublayerConnection,
    MultiHeadAttention, 
    PositionwiseFeedForward, 
    Decoder
)




class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.pff = PositionwiseFeedForward(config)
        self.sublayer = clones(SublayerConnection(config), 2)


    def forward(self, x, mask, apply_proj=False):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, apply_proj))
        return self.sublayer[1](x, self.pff)





class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()        

        self.attn = config.attn
        self.n_layers = config.n_layers
        self.embeddings = Embeddings(config)
        layer = EncoderLayer(config)
        self.layers = clones(layer, config.n_layers)


    def forward(self, x, e_mask):
        x = self.embeddings(x)

        if self.attn == 'orig':
            for layer in self.layers:
                x = layer(x, e_mask)
        else:
            if 'half' in self.attn:
                for idx, layer in enumerate(self.layers):
                    if (idx + 1) // 2 >= self.n_layers // 2:
                        x = layer(x, e_mask, apply_proj=True)
                    else:
                        x = layer(x, e_mask, apply_proj=False)                
            else:
                for layer in self.layers:
                    x = layer(x, e_mask, apply_proj=True)

        return x




class StandardTransformer(nn.Module):
    def __init__(self, config):
        super(StandardTransformer, self).__init__()

        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    def pad_mask(self, x):
        return (x != self.pad_id).unsqueeze(1).unsqueeze(2)


    def dec_mask(self, x):
        sz = x.size(1)
        pad_mask = self.pad_mask(x)
        sub_mask = torch.tril(torch.ones((sz, sz), device=self.device)).bool()
        return pad_mask & sub_mask


    @staticmethod
    def shift_y(x):
        return x[:, :-1], x[:, 1:]


    def forward(self, x, y):
        y, label = self.shift_y(y)

        e_mask = self.pad_mask(x) 
        d_mask = self.dec_mask(y)

        memory = self.encoder(x, e_mask)
        dec_out = self.decoder(y, memory, e_mask, d_mask)
        logit = self.generator(dec_out)

        #Getting Outputs
        self.out.logit = logit
        self.out.loss = self.criterion(
            logit.contiguous().view(-1, self.vocab_size), 
            label.contiguous().view(-1)
        )
        
        return self.out                