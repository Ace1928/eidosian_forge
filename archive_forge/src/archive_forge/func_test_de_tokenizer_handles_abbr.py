import pytest
@pytest.mark.parametrize('text', ['z.B.', 'd.h.', 'Jan.', 'Dez.', 'Chr.'])
def test_de_tokenizer_handles_abbr(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 1