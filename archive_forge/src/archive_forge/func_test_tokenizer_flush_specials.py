import re
import numpy
import pytest
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import (
from spacy.vocab import Vocab
def test_tokenizer_flush_specials(en_vocab):
    suffix_re = re.compile('[\\.]$')
    rules = {'a a': [{'ORTH': 'a a'}]}
    tokenizer1 = Tokenizer(en_vocab, suffix_search=suffix_re.search, rules=rules)
    assert [t.text for t in tokenizer1('a a.')] == ['a a', '.']
    tokenizer1.rules = {}
    assert [t.text for t in tokenizer1('a a.')] == ['a', 'a', '.']