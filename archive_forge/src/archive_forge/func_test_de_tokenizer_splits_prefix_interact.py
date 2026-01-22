import pytest
@pytest.mark.parametrize('text,length', [('z.B.', 1), ('zb.', 2), ('(z.B.', 2)])
def test_de_tokenizer_splits_prefix_interact(de_tokenizer, text, length):
    tokens = de_tokenizer(text)
    assert len(tokens) == length