import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.parametrize('string,symbol', [('IS_ALPHA', IS_ALPHA), ('NOUN', NOUN), ('VERB', VERB), ('LEMMA', LEMMA), ('ORTH', ORTH)])
def test_vocab_api_symbols(en_vocab, string, symbol):
    assert en_vocab.strings[string] == symbol