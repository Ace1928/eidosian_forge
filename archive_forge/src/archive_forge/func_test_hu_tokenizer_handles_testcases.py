import pytest
@pytest.mark.parametrize('text,expected_tokens', TESTS)
def test_hu_tokenizer_handles_testcases(hu_tokenizer, text, expected_tokens):
    tokens = hu_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list