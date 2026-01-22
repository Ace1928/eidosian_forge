import pytest
@pytest.mark.parametrize('text', ['αριθ.', 'τρισ.', 'δισ.', 'σελ.'])
def test_el_tokenizer_handles_abbr(el_tokenizer, text):
    tokens = el_tokenizer(text)
    assert len(tokens) == 1