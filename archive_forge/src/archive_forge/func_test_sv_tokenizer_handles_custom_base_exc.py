import pytest
def test_sv_tokenizer_handles_custom_base_exc(sv_tokenizer):
    text = 'Här är något du kan titta på.'
    tokens = sv_tokenizer(text)
    assert len(tokens) == 8
    assert tokens[6].text == 'på'
    assert tokens[7].text == '.'