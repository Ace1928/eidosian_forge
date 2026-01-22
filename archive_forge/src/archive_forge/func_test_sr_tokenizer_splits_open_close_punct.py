import pytest
@pytest.mark.parametrize('punct_open,punct_close', PUNCT_PAIRED)
@pytest.mark.parametrize('text', ['Тест'])
def test_sr_tokenizer_splits_open_close_punct(sr_tokenizer, punct_open, punct_close, text):
    tokens = sr_tokenizer(punct_open + text + punct_close)
    assert len(tokens) == 3
    assert tokens[0].text == punct_open
    assert tokens[1].text == text
    assert tokens[2].text == punct_close