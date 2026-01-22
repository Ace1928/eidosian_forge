import pytest
@pytest.mark.parametrize('text', ['janv.', 'juill.', 'Dr.', 'av.', 'sept.'])
def test_fr_tokenizer_handles_abbr(fr_tokenizer, text):
    tokens = fr_tokenizer(text)
    assert len(tokens) == 1