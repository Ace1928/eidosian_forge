import pytest
@pytest.mark.parametrize('text', ['(U.S.)'])
def test_en_tokenizer_splits_even_wrap_interact(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3