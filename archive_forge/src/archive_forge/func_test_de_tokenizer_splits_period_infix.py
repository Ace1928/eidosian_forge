import pytest
@pytest.mark.parametrize('text', ['blau.Rot', 'Hallo.Welt'])
def test_de_tokenizer_splits_period_infix(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 3