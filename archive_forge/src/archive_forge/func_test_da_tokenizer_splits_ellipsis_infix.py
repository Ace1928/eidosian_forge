import pytest
@pytest.mark.parametrize('text', ['sort...Gul', 'sort...gul'])
def test_da_tokenizer_splits_ellipsis_infix(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 3