import pytest
@pytest.mark.parametrize('text', ['z.B.)'])
def test_de_tokenizer_splits_suffix_interact(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 2