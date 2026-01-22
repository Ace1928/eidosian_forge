import pytest
@pytest.mark.parametrize('text', ['lorem ipsum'])
def test_tokenizer_splits_single_space(tokenizer, text):
    tokens = tokenizer(text)
    assert len(tokens) == 2