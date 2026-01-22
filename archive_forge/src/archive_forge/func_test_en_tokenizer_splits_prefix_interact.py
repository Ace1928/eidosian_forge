import pytest
@pytest.mark.parametrize('text,length', [('U.S.', 1), ('us.', 2), ('(U.S.', 2)])
def test_en_tokenizer_splits_prefix_interact(en_tokenizer, text, length):
    tokens = en_tokenizer(text)
    assert len(tokens) == length