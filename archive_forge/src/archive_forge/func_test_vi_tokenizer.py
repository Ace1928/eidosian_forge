import pytest
from spacy.lang.vi import Vietnamese
from ...tokenizer.test_naughty_strings import NAUGHTY_STRINGS
@pytest.mark.parametrize('text,expected_tokens', TOKENIZER_TESTS)
def test_vi_tokenizer(vi_tokenizer, text, expected_tokens):
    tokens = [token.text for token in vi_tokenizer(text)]
    assert tokens == expected_tokens