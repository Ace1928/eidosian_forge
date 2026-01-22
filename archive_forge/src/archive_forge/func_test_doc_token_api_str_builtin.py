import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.mark.parametrize('text', ['one two'])
def test_doc_token_api_str_builtin(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert str(tokens[0]) == text.split(' ')[0]
    assert str(tokens[1]) == text.split(' ')[1]