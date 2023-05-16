import re, os, json
from datasets import load_dataset



#Select and Tokenize Data
def process_data(orig_data):

    processed = []
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

        cnt += 1
        if cnt == volumn:
            break


    return processed



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