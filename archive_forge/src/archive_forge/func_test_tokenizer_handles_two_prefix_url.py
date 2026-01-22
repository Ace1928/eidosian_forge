import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.slow
@pytest.mark.parametrize('prefix1', PREFIXES)
@pytest.mark.parametrize('prefix2', PREFIXES)
@pytest.mark.parametrize('url', URLS_FULL)
def test_tokenizer_handles_two_prefix_url(tokenizer, prefix1, prefix2, url):
    tokens = tokenizer(prefix1 + prefix2 + url)
    assert len(tokens) == 3
    assert tokens[0].text == prefix1
    assert tokens[1].text == prefix2
    assert tokens[2].text == url