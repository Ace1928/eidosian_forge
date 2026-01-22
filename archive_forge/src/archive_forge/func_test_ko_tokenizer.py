import pytest
@pytest.mark.parametrize('text,expected_tokens', TOKENIZER_TESTS)
def test_ko_tokenizer(ko_tokenizer, text, expected_tokens):
    tokens = [token.text for token in ko_tokenizer(text)]
    assert tokens == expected_tokens.split()