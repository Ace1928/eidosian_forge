import pytest
@pytest.mark.parametrize('text,expected_tags', TAG_TESTS)
def test_ko_tokenizer_tags(ko_tokenizer, text, expected_tags):
    tags = [token.tag_ for token in ko_tokenizer(text)]
    assert tags == expected_tags.split()