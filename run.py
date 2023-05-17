import argparse, torch
import sentencepiece as spm
from module.test import Tester
from module.train import Trainer
from module.model import load_model
from module.data import load_dataloader
from transformers import set_seed, AutoTokenizer



class Config(object):
    def __init__(self, args):    

        self.mode = args.mode
        self.model_type = args.model

        tokenizer_dict = {'transformer_xl': "transfo-xl-wt103",
                          'reformer': "google/reformer-enwik8",
                          'longformer': "allenai/longformer-base-4096",
                          'bigbird': "google/bigbird-roberta-base"}
        self.tokenizer_name = tokenizer_dict[self.model_type]

        #Training args
        self.early_stop = True
        self.patience = 3        
        self.clip = 1
        self.lr = 5e-4
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4
        self.ckpt_path = f"ckpt/{self.model_type}.pt"

        #Model args
        self.n_heads = 8
        self.n_layers = 6
        self.pff_dim = 2048
        self.bert_dim = 768
        self.hidden_dim = 512
        self.dropout_ratio = 0.1
        self.model_max_length = 1024
        self.act = 'gelu'
        self.norm_first = True
        self.batch_first = True

        if self.mode == 'inference':
            self.search_method = args.search
            self.device = torch.device('cpu')
        else:
            self.search_method = None
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def inference(config, model, tokenizer):
    print('Type "quit" to terminate Summarization')
    
    while True:
        user_input = input('Please Type Text >> ')
        if user_input.lower() == 'quit':
            print('--- Terminate the Summarization ---')
            print('-' * 30)
            break

        src = config.src_tokenizer.Encode(user_input)
        src = torch.LongTensor(src).unsqueeze(0).to(config.device)

        if config.search == 'beam':
            pred_seq = config.search.beam_search(src)
        elif config.search == 'greedy':
            pred_seq = config.search.greedy_search(src)

        print(f" Original  Sequence: {user_input}")
        print(f'Summarized Sequence: {tokenizer.Decode(pred_seq)}\n')



def main(args):
    set_seed(42)
    config = Config(args)
    

    #Load Tokenizers
    enc_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    dec_tokenizer = spm.SentencePieceProcessor()
    dec_tokenizer.load(f'data/tokenizer.model')
    dec_tokenizer.SetEncodeExtraOptions('bos:eos')    

    setattr(config, 'vocab_size', dec_tokenizer.vocab_size())
    setattr(config, 'enc_pad_id', enc_tokenizer.pad_id())
    setattr(config, 'dec_pad_id', dec_tokenizer.pad_id())


    #Load model
    model = load_model(config)

    if config.mode == 'train': 
        train_dataloader = load_dataloader(config, enc_tokenizer, dec_tokenizer, 'train')
        valid_dataloader = load_dataloader(config, enc_tokenizer, dec_tokenizer, 'valid')
        trainer = Trainer(config, model, train_dataloader, valid_dataloader)
        trainer.train()
        return

    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, 'test')
        tester = Tester(config, model, enc_tokenizer, dec_tokenizer, test_dataloader)
        tester.test()
        return
    
    elif config.mode == 'inference':
        inference(config, model, enc_tokenizer, dec_tokenizer)
        return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode.lower() in ['train', 'test', 'inference']
    assert args.model.lower() in ['transformer_xl','reformer', 'longformer', 'bigbird']

    if args.task == 'inference':
        assert args.search in ['greedy', 'beam']

    main(args)