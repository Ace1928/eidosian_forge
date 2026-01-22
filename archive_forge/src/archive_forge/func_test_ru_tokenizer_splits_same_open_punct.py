from string import punctuation
import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Привет'])
def test_ru_tokenizer_splits_same_open_punct(ru_tokenizer, punct, text):
    tokens = ru_tokenizer(punct + punct + punct + text)
    assert len(tokens) == 4
    assert tokens[0].text == punct
    assert tokens[3].text == text