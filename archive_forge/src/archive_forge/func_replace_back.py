import re
from typing import Tuple
from nltk.stem.api import StemmerI
@staticmethod
def replace_back(word: str) -> str:
    word = Cistem.repl_xx_back.sub('\\1\\1', word)
    word = word.replace('%', 'ei')
    word = word.replace('&', 'ie')
    word = word.replace('$', 'sch')
    return word