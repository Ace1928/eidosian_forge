import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('text', ['Mabuhay'])
def test_tl_tokenizer_splits_same_close_punct(tl_tokenizer, punct, text):
    tokens = tl_tokenizer(text + punct + punct + punct)
    assert len(tokens) == 4
    assert tokens[0].text == text
    assert tokens[1].text == punct