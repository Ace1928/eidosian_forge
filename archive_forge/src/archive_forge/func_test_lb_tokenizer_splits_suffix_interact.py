import pytest
@pytest.mark.parametrize('text', ['z.B.)'])
def test_lb_tokenizer_splits_suffix_interact(lb_tokenizer, text):
    tokens = lb_tokenizer(text)
    assert len(tokens) == 2