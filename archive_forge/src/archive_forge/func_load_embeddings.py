import torch
import unicodedata
from collections import Counter
from parlai.core.build_data import modelzoo_path
def load_embeddings(opt, word_dict):
    """
    Initialize embeddings from file of pretrained vectors.
    """
    embeddings = torch.Tensor(len(word_dict), opt['embedding_dim'])
    embeddings.normal_(0, 1)
    opt['embedding_file'] = modelzoo_path(opt.get('datapath'), opt['embedding_file'])
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            if len(parsed) > 2:
                assert len(parsed) == opt['embedding_dim'] + 1
                w = normalize_text(parsed[0])
                if w in word_dict:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embeddings[word_dict[w]].copy_(vec)
    embeddings[word_dict['__NULL__']].fill_(0)
    return embeddings