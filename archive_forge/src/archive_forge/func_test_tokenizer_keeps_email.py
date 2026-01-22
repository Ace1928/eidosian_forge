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
@pytest.mark.parametrize('text', ['hello123@example.com', 'hi+there@gmail.it', 'matt@explosion.ai'])
def test_tokenizer_keeps_email(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 1