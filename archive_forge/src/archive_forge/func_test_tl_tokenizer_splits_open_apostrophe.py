import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('text', ["'Ang"])
def test_tl_tokenizer_splits_open_apostrophe(tl_tokenizer, text):
    tokens = tl_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "'"