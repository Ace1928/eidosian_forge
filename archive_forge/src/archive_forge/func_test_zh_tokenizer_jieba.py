import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
@pytest.mark.parametrize('text,expected_tokens', JIEBA_TOKENIZER_TESTS)
def test_zh_tokenizer_jieba(zh_tokenizer_jieba, text, expected_tokens):
    tokens = [token.text for token in zh_tokenizer_jieba(text)]
    assert tokens == expected_tokens