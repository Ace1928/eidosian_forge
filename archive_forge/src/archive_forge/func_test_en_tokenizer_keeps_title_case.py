import pytest
@pytest.mark.parametrize('pron', ['I', 'You', 'He', 'She', 'It', 'We', 'They'])
@pytest.mark.parametrize('contraction', ["'ll", "'d"])
def test_en_tokenizer_keeps_title_case(en_tokenizer, pron, contraction):
    tokens = en_tokenizer(pron + contraction)
    assert tokens[0].text == pron
    assert tokens[1].text == contraction