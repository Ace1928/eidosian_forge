import pytest
@pytest.mark.parametrize('text', ['halo...Malaysia', 'dia...pergi'])
def test_ms_tokenizer_splits_ellipsis_infix(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3