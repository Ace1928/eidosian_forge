import pytest
@pytest.mark.parametrize('text', ['Islam-Konferenz', 'Ost-West-Konflikt'])
def test_de_tokenizer_keeps_hyphens(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 1