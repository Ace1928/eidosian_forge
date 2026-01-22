import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def en_parser(en_vocab):
    nlp = get_lang_class('en')(en_vocab)
    return nlp.create_pipe('parser')