from abc import ABC
from abc import abstractmethod
from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
from typing import List, Union
@property
def sep(self):
    raise NotImplementedError('SEP is not provided for {} tokenizer'.format(self.name))