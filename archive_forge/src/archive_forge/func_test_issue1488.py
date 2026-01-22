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
@pytest.mark.issue(1488)
def test_issue1488():
    """Test that tokenizer can parse DOT inside non-whitespace separators"""
    prefix_re = re.compile('[\\[\\("\']')
    suffix_re = re.compile('[\\]\\)"\']')
    infix_re = re.compile('[-~\\.]')
    simple_url_re = re.compile('^https?://')

    def my_tokenizer(nlp):
        return Tokenizer(nlp.vocab, {}, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=simple_url_re.match)
    nlp = English()
    nlp.tokenizer = my_tokenizer(nlp)
    doc = nlp('This is a test.')
    for token in doc:
        assert token.text