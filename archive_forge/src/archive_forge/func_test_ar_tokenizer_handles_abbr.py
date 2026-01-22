import pytest
@pytest.mark.parametrize('text', ['ق.م', 'إلخ', 'ص.ب', 'ت.'])
def test_ar_tokenizer_handles_abbr(ar_tokenizer, text):
    tokens = ar_tokenizer(text)
    assert len(tokens) == 1