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
def test_tokenizer_initial_special_case_explain(en_vocab):
    tokenizer = Tokenizer(en_vocab, token_match=re.compile('^id$').match, rules={'id': [{'ORTH': 'i'}, {'ORTH': 'd'}]})
    tokens = [t.text for t in tokenizer('id')]
    explain_tokens = [t[1] for t in tokenizer.explain('id')]
    assert tokens == explain_tokens