import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.slow
@pytest.mark.parametrize('prefix', PREFIXES)
@pytest.mark.parametrize('url', URLS_FULL)
def test_tokenizer_handles_prefixed_url(tokenizer, prefix, url):
    tokens = tokenizer(prefix + url)
    assert len(tokens) == 2
    assert tokens[0].text == prefix
    assert tokens[1].text == url