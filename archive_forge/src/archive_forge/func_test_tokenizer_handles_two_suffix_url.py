import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.slow
@pytest.mark.parametrize('suffix1', SUFFIXES)
@pytest.mark.parametrize('suffix2', SUFFIXES)
@pytest.mark.parametrize('url', URLS_FULL)
def test_tokenizer_handles_two_suffix_url(tokenizer, suffix1, suffix2, url):
    tokens = tokenizer(url + suffix1 + suffix2)
    if suffix1 + suffix2 in BASE_EXCEPTIONS:
        assert len(tokens) == 2
        assert tokens[0].text == url
        assert tokens[1].text == suffix1 + suffix2
    else:
        assert len(tokens) == 3
        assert tokens[0].text == url
        assert tokens[1].text == suffix1
        assert tokens[2].text == suffix2