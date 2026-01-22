import pytest
@pytest.mark.parametrize('text,expected_tokens', TESTCASES)
def test_bn_tokenizer_handles_testcases(bn_tokenizer, text, expected_tokens):
    tokens = bn_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list