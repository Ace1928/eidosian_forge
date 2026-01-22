import pytest
@pytest.mark.parametrize('text,norm', [('Jan.', 'January'), ("'cuz", 'because')])
def test_en_lex_attrs_norm_exceptions(en_tokenizer, text, norm):
    tokens = en_tokenizer(text)
    assert tokens[0].norm_ == norm