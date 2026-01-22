import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def de_vocab():
    return get_lang_class('de')().vocab