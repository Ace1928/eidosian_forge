import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.slow
@pytest.mark.parametrize('prefix', PREFIXES)
@pytest.mark.parametrize('suffix', SUFFIXES)
@pytest.mark.parametrize('url', URLS_FULL)
def test_tokenizer_handles_surround_url(tokenizer, prefix, suffix, url):
    tokens = tokenizer(prefix + url + suffix)
    assert len(tokens) == 3
    assert tokens[0].text == prefix
    assert tokens[1].text == url
    assert tokens[2].text == suffix