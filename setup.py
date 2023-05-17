import re, os, json
import sentencepiece as spm
from datasets import load_dataset



#Select and Tokenize Data
def process_data(orig_data):

    processed, corpus = [], []
    cnt, volumn = 0, 12000
    min_len, max_len = 1000, 3000

    
    for elem in orig_data:
        text, summ = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(text) < max_len):
            continue

        #remove unnecessary characters in trg sequence
        summ = re.sub(r'\n', ' ', summ)                 #remove \n
        summ = re.sub(r"\s([.](?:\s|$))", r'\1', summ)  #remove whitespace in front of dot
        
        processed.append({"text": text, 'summ': summ})
        corpus.append(summ)

        cnt += 1
        if cnt == volumn:
            break

    with open('data/corpus.txt', 'w') as f:
        json.dump(corpus, f)

    return processed



def build_vocab():
    assert os.path.exists(f'data/corpus.txt')
    opt = f"--input=data/corpus.txt \
            --model_prefix=data/tokenizer \
            --vocab_size=30000 \
            --character_coverage=1 \
            --model_type=bpe \
            --pad_id=0 --pad_piece=[PAD] \
            --unk_id=1 --unk_piece=[UNK] \
            --bos_id=2 --bos_piece=[BOS] \
            --eos_id=3 --eos_piece=[EOS]".replace(' '*12, '')

    spm.SentencePieceTrainer.Train(opt)
    os.remove('data/corpus.txt')



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)                    



def main():
    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    processed = process_data(orig)    
    save_data(processed)


if __name__ == '__main__':
    main()