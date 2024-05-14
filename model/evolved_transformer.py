import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from collections import namedtuple
from .components import (
    clones, 
    Embeddings, 
    Decoder, 
    MultiHeadAttention
)




#GLU
class GatedConvolution(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, padding=1):
        super(GatedConvolution,self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim * 2,
            kernel_size=kernel_size, 
            padding=padding, bias=True
        )

        init.xavier_uniform_(self.conv.weight, gain=1)

    def forward(self,x):
        convoluted = self.conv(x.transpose(1,2)).transpose(1,2)
        out, gate = convoluted.split(int(convoluted.size(-1) / 2), -1)
        out = out * torch.sigmoid(gate)
        return out




class SeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv1D, self).__init__()

        self.depth_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels
        )
        
        self.point_wise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out




class EncoderCell(nn.Module):
    def __init__(self, config):
        super(EncoderCell, self).__init__()

        self.pad_id = config.pad_id
        self.glu = GatedConvolution(config.hidden_dim)
        self.attention = MultiHeadAttention(config)

        self.mid_layer_norm = nn.LayerNorm(config.pff_dim)
        self.layer_norms = clones(nn.LayerNorm(config.hidden_dim), 4)

        self.left_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_ratio)
        )

        self.right_net = nn.Sequential(
            nn.Conv1d(in_channels=config.hidden_dim, 
                      out_channels=config.hidden_dim//2, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(config.dropout_ratio)
        )

        self.sep_conv = SeparableConv1D(
            config.pff_dim, config.hidden_dim // 2, 9
        )

        self.pff = nn.Sequential(
            nn.Linear(config.hidden_dim, config.pff_dim),
            nn.SiLU(),
            nn.Linear(config.pff_dim, config.hidden_dim)
        )


    def forward(self, x, e_mask, apply_proj=False):
        ### Block_01
        B01_out = self.glu(self.layer_norms[0](x)) #Dim:512


        ### Block_02
        B02_normed = self.layer_norms[1](B01_out)        

        left_out = self.left_net(B02_normed)
        right_out = self.right_net(B02_normed.transpose(1, 2)).transpose(1, 2)

        right_out = F.pad(
            input=right_out, 
            pad=(0, left_out.size(-1) - right_out.size(-1), 0,0,0,0), 
            mode='constant', value=self.pad_id
        ) #Dim:2048          

        B02_out = left_out + right_out


        ### Block_03
        B03_out = self.mid_layer_norm(B02_out)
        
        B03_out = self.sep_conv(
            B03_out.transpose(1, 2)
        ).transpose(1, 2) #Dim:256
        
        B03_out = F.pad(
            input=B03_out,
            pad=(0, B01_out.size(-1) - B03_out.size(-1), 0, 0, 0, 0),
            mode='constant', value=self.pad_id
        )
        
        B03_out += B01_out #Dim:512


        ### Block_04
        B04_out = self.layer_norms[2](B03_out)
        
        attention_out = self.attention(
            B04_out, B04_out, B04_out,
            mask = e_mask, apply_proj=apply_proj
        )
        
        B04_out += attention_out #Dim:512


        ### Block_05 & 06
        out = self.layer_norms[3](B04_out)
        out = self.pff(out) + B04_out #Dim:512
        return out 




class EvolvedEncoder(nn.Module):
    def __init__(self, config):
        super(EvolvedEncoder, self).__init__()

        self.attn = config.attn
        self.n_cells = config.n_layers//2

        self.embeddings = Embeddings(config)
        self.cells = clones(EncoderCell(config), self.n_cells)


    def forward(self, x, e_mask):
        x = self.embeddings(x)

        if self.attn == 'orig':
            for cell in self.cells:
                x = cell(x, e_mask)
        else:
            if 'half' in self.attn:
                for idx, cell in enumerate(self.cells):
                    if (idx + 1) // 2 >= self.n_cells // 2:
                        x = cell(x, e_mask, apply_proj=True)
                    else:
                        x = cell(x, e_mask, apply_proj=False)                
            else:
                for cell in self.cells:
                    x = cell(x, e_mask, apply_proj=True)

        return x




class EvolvedTransformer(nn.Module):
    def __init__(self, config):
        super(EvolvedTransformer, self).__init__()
        
        self.pad_id = config.pad_id
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = EvolvedEncoder(config)
        self.decoder = Decoder(config)
        self.generator = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss()
        self.out = namedtuple('Out', 'logit loss')


    @staticmethod    
    def shift_y(x):
        return x[:, :-1], x[:, 1:]    


    def pad_mask(self, x):
        return x == self.pad_id

    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


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
