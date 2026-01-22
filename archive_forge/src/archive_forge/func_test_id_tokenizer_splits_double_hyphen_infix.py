import pytest
def test_id_tokenizer_splits_double_hyphen_infix(id_tokenizer):
    tokens = id_tokenizer('Arsene Wenger--manajer Arsenal--melakukan konferensi pers.')
    assert len(tokens) == 10
    assert tokens[0].text == 'Arsene'
    assert tokens[1].text == 'Wenger'
    assert tokens[2].text == '--'
    assert tokens[3].text == 'manajer'
    assert tokens[4].text == 'Arsenal'
    assert tokens[5].text == '--'
    assert tokens[6].text == 'melakukan'
    assert tokens[7].text == 'konferensi'
    assert tokens[8].text == 'pers'
    assert tokens[9].text == '.'