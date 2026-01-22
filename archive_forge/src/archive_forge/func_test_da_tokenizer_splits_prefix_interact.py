import pytest
@pytest.mark.parametrize('text,expected', [('f.eks.', ['f.eks.']), ('fe.', ['fe', '.']), ('(f.eks.', ['(', 'f.eks.'])])
def test_da_tokenizer_splits_prefix_interact(da_tokenizer, text, expected):
    tokens = da_tokenizer(text)
    assert len(tokens) == len(expected)
    assert [t.text for t in tokens] == expected