import json
import os
import re
import sys
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
def ja_tokenize(self, text):
    if self.ja_word_tokenizer is None:
        try:
            import Mykytea
            self.ja_word_tokenizer = Mykytea.Mykytea(f'-model {os.path.expanduser('~')}/local/share/kytea/model.bin')
        except (AttributeError, ImportError):
            logger.error("Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper (https://github.com/chezou/Mykytea-python) with the following steps")
            logger.error('1. git clone git@github.com:neubig/kytea.git && cd kytea')
            logger.error('2. autoreconf -i')
            logger.error('3. ./configure --prefix=$HOME/local')
            logger.error('4. make && make install')
            logger.error('5. pip install kytea')
            raise
    return list(self.ja_word_tokenizer.getWS(text))