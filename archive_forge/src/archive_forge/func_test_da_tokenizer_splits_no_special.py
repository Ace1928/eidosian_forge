import pytest
@pytest.mark.parametrize('text', ['(under)'])
def test_da_tokenizer_splits_no_special(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 3