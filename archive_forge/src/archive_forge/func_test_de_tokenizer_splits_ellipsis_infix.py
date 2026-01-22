import pytest
@pytest.mark.parametrize('text', ['blau...Rot', 'blau...rot'])
def test_de_tokenizer_splits_ellipsis_infix(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 3