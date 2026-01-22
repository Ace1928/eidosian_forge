import pytest
@pytest.mark.parametrize('text', ["'Тест"])
def test_uk_tokenizer_splits_open_appostrophe(uk_tokenizer, text):
    tokens = uk_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "'"