import os
import pytest
from spacy.attrs import IS_ALPHA, LEMMA, ORTH
from spacy.lang.en import English
from spacy.parts_of_speech import NOUN, VERB
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_to_disk_exclude():
    nlp = English()
    with make_tempdir() as d:
        nlp.vocab.to_disk(d, exclude=('vectors', 'lookups'))
        assert 'vectors' not in os.listdir(d)
        assert 'lookups.bin' not in os.listdir(d)