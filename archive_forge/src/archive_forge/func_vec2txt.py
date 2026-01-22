from parlai.core.dict import DictionaryAgent
from parlai.zoo.bert.build import download
from .helpers import VOCAB_PATH
import os
def vec2txt(self, vec):
    if not isinstance(vec, list):
        idxs = [idx.item() for idx in vec.cpu()]
    else:
        idxs = vec
    toks = self.tokenizer.convert_ids_to_tokens(idxs)
    return ' '.join(toks)