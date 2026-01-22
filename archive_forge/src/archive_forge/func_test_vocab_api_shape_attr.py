import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('text', ['example'])
def test_vocab_api_shape_attr(en_vocab, text):
    lex = en_vocab[text]
    assert lex.orth != lex.shape