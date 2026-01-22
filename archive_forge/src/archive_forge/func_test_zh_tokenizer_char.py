import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
@pytest.mark.parametrize('text', TEXTS)
def test_zh_tokenizer_char(zh_tokenizer_char, text):
    tokens = [token.text for token in zh_tokenizer_char(text)]
    assert tokens == list(text)