import pytest
@pytest.mark.parametrize('text', ["gitta'r", "Björn's", "Lars'"])
def test_tokenizer_handles_no_punct(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 1