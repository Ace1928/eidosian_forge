import pytest
@pytest.mark.parametrize('punct', PUNCT_OPEN)
@pytest.mark.parametrize('text', ['Здраво'])
def test_sr_tokenizer_splits_open_punct(sr_tokenizer, punct, text):
    tokens = sr_tokenizer(punct + text)
    assert len(tokens) == 2
    assert tokens[0].text == punct
    assert tokens[1].text == text