import os, re, json, yaml
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents





#Summarization
def process_data(data_volumn):    
    volumn_cnt = 0
    corpus, processed = [], []
    min_len, max_len = 500, 2300

    #Load Original Dataset
    cnn_data = load_dataset('cnn_dailymail', '3.0.0')

    for split in ['train', 'validation', 'test']:
        for elem in cnn_data[split]:

            x, y = elem['article'], elem['highlights']

            if min_len < len(x) < max_len:
                if len(y) < min_len:
                    
                    #Lowercase
                    x, y = x.lower(), y.lower()

                    #Remove unnecessary characters in trg sequence
                    y = re.sub(r'\n', ' ', y)                 #remove \n
                    y = re.sub(r"\s([.](?:\s|$))", r'\1', y)  #remove whitespace in front of dot

                    processed.append({'x': x, 'y': y})
                    corpus.append(x)
                    corpus.append(y)

                    #End Condition
                    volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return processed           



def train_tokenizer():
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)
    
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    tokenizer = Tokenizer(BPE(unk_token=vocab_config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_config['vocab_size'], 
        special_tokens=[
            vocab_config['pad_token'], 
            vocab_config['unk_token'],
            vocab_config['bos_token'],
            vocab_config['eos_token']
            ]
        )

    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(f"data/tokenizer.json")



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def main():
    data_volumn = 55100

    processed = process_data(data_volumn)
    train_tokenizer()
    save_data(processed)



if __name__ == '__main__':
    main()
