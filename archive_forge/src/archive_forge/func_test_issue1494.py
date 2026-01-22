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
@pytest.mark.issue(1494)
def test_issue1494():
    """Test if infix_finditer works correctly"""
    infix_re = re.compile('[^a-z]')
    test_cases = [('token 123test', ['token', '1', '2', '3', 'test']), ('token 1test', ['token', '1test']), ('hello...test', ['hello', '.', '.', '.', 'test'])]

    def new_tokenizer(nlp):
        return Tokenizer(nlp.vocab, {}, infix_finditer=infix_re.finditer)
    nlp = English()
    nlp.tokenizer = new_tokenizer(nlp)
    for text, expected in test_cases:
        assert [token.text for token in nlp(text)] == expected