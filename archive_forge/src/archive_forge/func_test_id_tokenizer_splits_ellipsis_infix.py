import pytest
@pytest.mark.parametrize('text', ['halo...Bandung', 'dia...pergi'])
def test_id_tokenizer_splits_ellipsis_infix(id_tokenizer, text):
    tokens = id_tokenizer(text)
    assert len(tokens) == 3