import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def nl_vocab():
    return get_lang_class('nl')().vocab