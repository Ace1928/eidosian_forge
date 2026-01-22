import pytest
@pytest.mark.parametrize('text,expected_tokens', HYPHENATED_TESTS)
def test_fi_tokenizer_hyphenated_words(fi_tokenizer, text, expected_tokens):
    tokens = fi_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list