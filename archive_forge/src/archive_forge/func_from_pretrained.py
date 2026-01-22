import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
@classmethod
@torch_only_method
def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
    """
        Instantiate a pre-processed corpus.
        """
    vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    is_local = os.path.isdir(pretrained_model_name_or_path)
    try:
        resolved_corpus_file = cached_file(pretrained_model_name_or_path, CORPUS_NAME, cache_dir=cache_dir)
    except EnvironmentError:
        logger.error(f"Corpus '{pretrained_model_name_or_path}' was not found in corpus list ({', '.join(PRETRAINED_CORPUS_ARCHIVE_MAP.keys())}. We assumed '{pretrained_model_name_or_path}' was a path or url but couldn't find files {CORPUS_NAME} at this path or url.")
        return None
    if is_local:
        logger.info(f'loading corpus file {resolved_corpus_file}')
    else:
        logger.info(f'loading corpus file {CORPUS_NAME} from cache at {resolved_corpus_file}')
    corpus = cls(*inputs, **kwargs)
    corpus_dict = torch.load(resolved_corpus_file)
    for key, value in corpus_dict.items():
        corpus.__dict__[key] = value
    corpus.vocab = vocab
    if corpus.train is not None:
        corpus.train = torch.tensor(corpus.train, dtype=torch.long)
    if corpus.valid is not None:
        corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)
    if corpus.test is not None:
        corpus.test = torch.tensor(corpus.test, dtype=torch.long)
    return corpus