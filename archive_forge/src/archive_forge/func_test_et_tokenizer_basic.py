import pytest
@pytest.mark.parametrize('text,expected_tokens', ET_BASIC_TOKENIZATION_TESTS)
def test_et_tokenizer_basic(et_tokenizer, text, expected_tokens):
    tokens = et_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list