import pytest
def test_ms_tokenizer_splits_double_hyphen_infix(id_tokenizer):
    tokens = id_tokenizer('Arsene Wenger--pengurus Arsenal--mengadakan sidang media.')
    assert len(tokens) == 10
    assert tokens[0].text == 'Arsene'
    assert tokens[1].text == 'Wenger'
    assert tokens[2].text == '--'
    assert tokens[3].text == 'pengurus'
    assert tokens[4].text == 'Arsenal'
    assert tokens[5].text == '--'
    assert tokens[6].text == 'mengadakan'
    assert tokens[7].text == 'sidang'
    assert tokens[8].text == 'media'
    assert tokens[9].text == '.'