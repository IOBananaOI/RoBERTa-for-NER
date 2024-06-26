import torch
from torch.utils.data import Dataset
from datasets import Dataset


class CONLL(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer):
        super(CONLL, self).__init__()

        self.max_len = 124
        
        self.tokenizer = tokenizer
        
        self.data = dataset

    def __getitem__(self, index):
        
        tokens, tags = self.data[index]['tokens'], self.data[index]['ner_tags']
            
        tokens = self.tokenizer.encode(tokens, is_split_into_words=True, max_length=self.max_len, padding='max_length', truncation=True)
        
        tags = [-100] + tags + [-100] * (len(tokens) - len(tags) - 1)
        
        tags = tags[:len(tokens)]
        
        return tokens, tags

    def __len__(self):
        return len(self.data)