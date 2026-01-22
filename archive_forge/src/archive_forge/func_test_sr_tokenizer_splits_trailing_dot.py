import pytest
@pytest.mark.parametrize('text', ['Тест.'])
def test_sr_tokenizer_splits_trailing_dot(sr_tokenizer, text):
    tokens = sr_tokenizer(text)
    assert tokens[1].text == '.'