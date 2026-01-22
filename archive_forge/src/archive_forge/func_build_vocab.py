import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def build_vocab(self):
    if self.vocab_file:
        logger.info(f'building vocab from {self.vocab_file}')
        self._build_from_file(self.vocab_file)
        logger.info(f'Final vocab size {len(self.sym2idx)}')
    else:
        logger.info(f'building vocab with min_freq={self.min_freq}, max_size={self.max_size}')
        self.idx2sym = []
        self.sym2idx = OrderedDict()
        for sym in self.special:
            self.add_special(sym)
        for sym, cnt in self.counter.most_common(self.max_size):
            if cnt < self.min_freq:
                break
            self.add_symbol(sym)
        logger.info(f'Final vocab size {len(self.sym2idx)} from {len(self.counter)} unique tokens')