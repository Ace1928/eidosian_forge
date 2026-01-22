import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
@pytest.mark.parametrize('text,expected_tokens', PKUSEG_TOKENIZER_TESTS)
def test_zh_tokenizer_pkuseg(zh_tokenizer_pkuseg, text, expected_tokens):
    tokens = [token.text for token in zh_tokenizer_pkuseg(text)]
    assert tokens == expected_tokens