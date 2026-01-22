import pytest
@pytest.mark.parametrize('text', ['S.Kom.)'])
def test_ms_tokenizer_splits_suffix_interact(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 2