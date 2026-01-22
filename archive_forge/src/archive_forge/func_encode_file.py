import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
@torch_only_method
def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
    if verbose:
        logger.info(f'encoding file {path} ...')
    assert os.path.exists(path), f'Output file {path} not found'
    encoded = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if verbose and idx > 0 and (idx % 500000 == 0):
                logger.info(f'    line {idx}')
            symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)
            encoded.append(self.convert_to_tensor(symbols))
    if ordered:
        encoded = torch.cat(encoded)
    return encoded