import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture
def hu_tokenizer():
    return get_lang_class('hu')().tokenizer