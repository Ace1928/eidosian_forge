import pytest
@pytest.mark.parametrize('text,expected_tags', FULL_TAG_TESTS)
def test_ko_tokenizer_full_tags(ko_tokenizer, text, expected_tags):
    tags = ko_tokenizer(text).user_data['full_tags']
    assert tags == expected_tags.split()