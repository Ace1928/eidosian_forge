import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def ru_lookup_lemmatizer():
    pytest.importorskip('pymorphy3')
    return get_lang_class('ru')().add_pipe('lemmatizer', config={'mode': 'pymorphy3_lookup'})