import pytest
@pytest.mark.parametrize('text', ['svart.Gul', 'Hej.Världen'])
def test_tokenizer_splits_period_infix(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 3