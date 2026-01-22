import pytest
from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
@pytest.mark.parametrize('url', URLS_BASIC)
def test_tokenizer_handles_simple_surround_url(tokenizer, url):
    tokens = tokenizer('(' + url + ')')
    assert len(tokens) == 3
    assert tokens[0].text == '('
    assert tokens[1].text == url
    assert tokens[2].text == ')'