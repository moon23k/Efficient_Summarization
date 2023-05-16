import os, torch
import torch.nn as nn
from collections import namedtuple
from transformers import (TransformerXLModel, TransformerXLConfig, 
                          ReformerModel, ReformerConfig, 
                          LongformerModel, LongformerConfig 
                          BigbirdModel, BigbirdConfig)



class PositionalEncoding(nn.Module):
    def __init__(self, config, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_ratio)
        
        pe = torch.zeros(max_len, config.emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.emb_dim, 2) * -(math.log(10000.0) / config.emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.scale = math.sqrt(config.emb_dim)

        self.pos_emb = PositionalEncoding(config)
        self.fc = nn.Linear(config.emb_dim, config.hidden_dim)

    def forward(self, x):
        out = self.tok_emb(x) * self.scale
        out = self.pos_emb(out)
        return self.fc(out)



class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.embeddings = Embeddings(config)
        
        layer = nn.TransformerDecoderLayer(d_model=config.hidden_dim,
                                           nhead=config.n_heads,
                                           dim_feedforward=config.pff_dim,
                                           dropout=config.dropout_ratio,
                                           batch_first=True, norm_first=True)
        
        self.decoder = nn.TransformerDecoder(decoder_layer=layer, 
                                             num_layers=config.n_layers,
                                             norm=nn.LayerNorm(config.hidden_dim))
        

    def forward(self, x, memory, x_sub_mask, x_pad_mask, m_pad_mask):
        return self.decoder(self.embeddings(x), memory, 
                            memory_key_padding_mask=m_pad_mask, 
                            tgt_key_padding_mask=x_pad_mask, 
                            tgt_mask=x_sub_mask)




def SparseModel(nn.Module):
    def __init__(self, config, encoder):
        self.device = config.device
        self.vocab_size = config.vocab_size

        self.encoder = encoder
        self.decoder = Decoder(config)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.classifier = nn.Linear(config.hidden_dim, config.vocab_size)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.out = namedtuple('Out', 'logit loss')


    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask, labels):
        memory = self.encoder(input_ids=enc_ids, 
                              attention_mask=enc_mask).last_hidden_state

        dec_out = self.decoder(dec_ids, memory, enc_mask, dec_mask)
        logit = self.classifier(self.dropout(dec_out))
        loss = self.criterion(logit.contiguous.view(-1, self.vocab_size),
                              labels.contiguous.view(-1))

        return self.out(logit, loss)




def init_weights(model):    
    for name, param in model.named_parameters():
        if any([x in name for x in ['embeddings', 'norm', 'bias']]):
            continue
        nn.init.xavier_uniform_(param)    


def print_model_desc(model):
    def count_params(model):
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return params

    def check_size(model):
        param_size, buffer_size = 0, 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")

    


def load_model(config):
    #Load Sparse Attention Transformer Based PLM Model
    if config.model_type == 'transformer_xl':
        model_config = TransformerXLConfig()
        encoder = TransformerXLModel(model_config)
    
    elif config.model_type == 'reformer':
        model_config = ReformerConfig()
        encoder = ReformerModel(model_config)

    elif config.model_type == 'longformer':
        model_config = LongformerConfig()
        encoder = LongformerModel(model_config)
    
    elif config.model_type == 'bigbird':
        model_config = BigbirdConfig()
        encoder = BigbirdModel(model_config)        

    model = SparseModel(config, encoder)

    init_weights(model)
    print(f'{config.strategy.upper()} Model has Loaded')

    if config.mode != 'train':
        ckpt = config.ckpt
        assert os.path.exists(ckpt)
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model States has loaded from {ckpt}")

    print_model_desc(model)
    return model.to(config.device)