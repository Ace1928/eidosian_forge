from string import punctuation
import pytest
@pytest.mark.parametrize('punct', PUNCT_CLOSE)
@pytest.mark.parametrize('text', ['Привет'])
def test_ru_tokenizer_splits_close_punct(ru_tokenizer, punct, text):
    tokens = ru_tokenizer(text + punct)
    assert len(tokens) == 2
    assert tokens[0].text == text
    assert tokens[1].text == punct