import re
import string
import hypothesis
import hypothesis.strategies
import pytest
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import get_lang_class
def test_tokenizer_explain_special_matcher_whitespace(en_vocab):
    rules = {':]': [{'ORTH': ':]'}]}
    tokenizer = Tokenizer(en_vocab, rules=rules)
    text = ': ]'
    tokens = [t.text for t in tokenizer(text)]
    explain_tokens = [t[1] for t in tokenizer.explain(text)]
    assert tokens == explain_tokens