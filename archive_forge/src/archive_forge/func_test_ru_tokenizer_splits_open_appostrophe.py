from string import punctuation
import pytest
@pytest.mark.parametrize('text', ["'Тест"])
def test_ru_tokenizer_splits_open_appostrophe(ru_tokenizer, text):
    tokens = ru_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "'"