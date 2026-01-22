import pytest
@pytest.mark.parametrize('text,length', [('z.B.', 1), ('zb.', 2), ('(z.B.', 2)])
def test_lb_tokenizer_splits_prefix_interact(lb_tokenizer, text, length):
    tokens = lb_tokenizer(text)
    assert len(tokens) == length