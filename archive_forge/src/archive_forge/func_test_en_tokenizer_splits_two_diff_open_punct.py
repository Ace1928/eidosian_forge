import pytest
from spacy.lang.punctuation import TOKENIZER_PREFIXES
from spacy.util import compile_prefix_regex
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('punct_add', ['`'])
@pytest.mark.parametrize('text', ['Hello'])
def test_en_tokenizer_splits_two_diff_open_punct(en_tokenizer, punct, punct_add, text):
    tokens = en_tokenizer(punct + punct_add + text)
    assert len(tokens) == 3
    assert tokens[0].text == punct
    assert tokens[1].text == punct_add
    assert tokens[2].text == text