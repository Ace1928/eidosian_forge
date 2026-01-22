import pytest
@pytest.mark.parametrize('text,expected_tokens', SPACY_TOKENIZER_TESTS)
def test_ko_spacy_tokenizer(ko_tokenizer_tokenizer, text, expected_tokens):
    tokens = [token.text for token in ko_tokenizer_tokenizer(text)]
    assert tokens == expected_tokens.split()