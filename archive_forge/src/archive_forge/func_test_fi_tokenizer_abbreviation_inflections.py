import pytest
@pytest.mark.parametrize('text,expected_tokens', ABBREVIATION_INFLECTION_TESTS)
def test_fi_tokenizer_abbreviation_inflections(fi_tokenizer, text, expected_tokens):
    tokens = fi_tokenizer(text)
    token_list = [token.text for token in tokens if not token.is_space]
    assert expected_tokens == token_list