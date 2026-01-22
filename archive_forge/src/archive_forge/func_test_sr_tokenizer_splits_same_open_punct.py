import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Здраво'])
def test_sr_tokenizer_splits_same_open_punct(sr_tokenizer, punct, text):
    tokens = sr_tokenizer(punct + punct + punct + text)
    assert len(tokens) == 4
    assert tokens[0].text == punct
    assert tokens[3].text == text