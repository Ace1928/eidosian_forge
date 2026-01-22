import pytest
@pytest.mark.parametrize('text,expected_pos', POS_TESTS)
def test_ko_tokenizer_pos(ko_tokenizer, text, expected_pos):
    pos = [token.pos_ for token in ko_tokenizer(text)]
    assert pos == expected_pos.split()