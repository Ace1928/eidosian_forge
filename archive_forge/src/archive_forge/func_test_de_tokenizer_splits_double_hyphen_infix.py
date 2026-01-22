import pytest
def test_de_tokenizer_splits_double_hyphen_infix(de_tokenizer):
    tokens = de_tokenizer('Viele Regeln--wie die Bindestrich-Regeln--sind kompliziert.')
    assert len(tokens) == 10
    assert tokens[0].text == 'Viele'
    assert tokens[1].text == 'Regeln'
    assert tokens[2].text == '--'
    assert tokens[3].text == 'wie'
    assert tokens[4].text == 'die'
    assert tokens[5].text == 'Bindestrich-Regeln'
    assert tokens[6].text == '--'
    assert tokens[7].text == 'sind'
    assert tokens[8].text == 'kompliziert'