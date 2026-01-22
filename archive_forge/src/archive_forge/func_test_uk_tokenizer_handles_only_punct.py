import pytest
@pytest.mark.parametrize('text', ['(', '((', '<'])
def test_uk_tokenizer_handles_only_punct(uk_tokenizer, text):
    tokens = uk_tokenizer(text)
    assert len(tokens) == len(text)