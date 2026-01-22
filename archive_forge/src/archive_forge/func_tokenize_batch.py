from abc import ABC
from abc import abstractmethod
from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
from typing import List, Union
def tokenize_batch(self, text_batch: Union[List[str], str]):
    return self.tokenizer.encode_batch(text_batch)