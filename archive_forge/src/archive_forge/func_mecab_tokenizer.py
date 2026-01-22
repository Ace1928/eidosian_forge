from typing import Any, Dict, Iterator
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...symbols import POS, X
from ...tokens import Doc
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_INFIXES
from .stop_words import STOP_WORDS
from .tag_map import TAG_MAP
@property
def mecab_tokenizer(self):
    if self._mecab_tokenizer is None:
        self._mecab_tokenizer = self._mecab('-F%f[0],%f[7]')
    return self._mecab_tokenizer