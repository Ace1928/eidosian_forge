def test_en_simple_punct(en_tokenizer):
    text = 'to walk, do foo'
    tokens = en_tokenizer(text)
    assert tokens[0].idx == 0
    assert tokens[1].idx == 3
    assert tokens[2].idx == 7
    assert tokens[3].idx == 9
    assert tokens[4].idx == 12