import pytest
@pytest.mark.parametrize('text,expected_tokens', AF_BASIC_TOKENIZATION_TESTS)
def test_af_tokenizer_basic(af_tokenizer, text, expected_tokens):
    tokens = af_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list