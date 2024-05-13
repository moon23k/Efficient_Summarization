import os, re, json, yaml
from run import load_tokenizer
from datasets import load_dataset
from tokenizers.models import BPE
from tokenizers import Tokenizer, normalizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents




def select_data(data_volumn):
    volumn_cnt = 0
    corpus, selected = [], []
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

                    selected.append({'x': x, 'y': y})
                    corpus.append(x)
                    corpus.append(y)

                    #End Condition
                    volumn_cnt += 1
            if volumn_cnt == data_volumn:
                break

    with open('data/corpus.txt', 'w') as f:
        f.write('\n'.join(corpus))
    
    return selected




def train_tokenizer(config):
    corpus_path = f'data/corpus.txt'
    assert os.path.exists(corpus_path)

    #Setting Tokenizer
    tokenizer = Tokenizer(BPE(unk_token=config['unk_token']))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=config['vocab_size'], 
        special_tokens=[
            config['pad_token'], config['unk_token'],
            config['bos_token'], config['eos_token']
            ]
        )

    #Train and Save Tokenizer
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer_path = "data/tokenizer.json"
    tokenizer.save(tokenizer_path)

    #Load Trained Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config['bos_token']} $A {config['eos_token']}",
        special_tokens=[(config['bos_token'], config['bos_id']), 
                        (config['eos_token'], config['eos_id'])]
        )

    return tokenizer




def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-5100], data_obj[-5100:-100], data_obj[-100:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{key}.json')




def process_data(tokenizer, selected_data):
    processed = []
    max_seq_len = 0

    for elem in selected_data:
        x = tokenizer(elem['x']).ids
        y = tokenizer(elem['y']).ids

        processed.append({'x': x, 'y': y})

        seq_len = len(x)
        if max_seq_len < seq_len:
            max_seq_len = seq_len


    return processed, max_seq_len




def main():
    with open("config.yaml", "r") as f:
        data_config = yaml.safe_load(f)['tokenizer']

    selected_data = select_data(data_config['data_volumn'])
    tokenizer = train_tokenizer(data_config)
    processed_data, max_seq_len = process_data(tokenizer, selected_data)
    save_data(processed_data)


    #Updata data configuration
    data_config.max_seq_len = max_seq_len
    with open("config.yaml", "w") as f:
        yaml.safe_dump(data_config, f)




if __name__ == '__main__':
    main()
