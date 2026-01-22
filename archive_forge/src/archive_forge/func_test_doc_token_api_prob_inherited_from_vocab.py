import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.mark.parametrize('text', ['Give it back! He pleaded.'])
def test_doc_token_api_prob_inherited_from_vocab(en_tokenizer, text):
    word = text.split()[0]
    en_tokenizer.vocab[word].prob = -1
    tokens = en_tokenizer(text)
    assert tokens[0].prob != 0