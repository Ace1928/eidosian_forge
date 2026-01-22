import pytest
@pytest.mark.parametrize('text', ["(ta'r?)"])
def test_da_tokenizer_splits_uneven_wrap(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 4
    assert tokens[0].text == '('
    assert tokens[1].text == "ta'r"
    assert tokens[2].text == '?'
    assert tokens[3].text == ')'