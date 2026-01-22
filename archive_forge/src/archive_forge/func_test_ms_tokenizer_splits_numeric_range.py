import pytest
@pytest.mark.parametrize('text', ['0.1-13.5', '0.0-0.1', '103.27-300'])
def test_ms_tokenizer_splits_numeric_range(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3