import pytest
@pytest.mark.parametrize('text', ["(ta'r"])
def test_da_tokenizer_splits_prefix_punct(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == '('
    assert tokens[1].text == "ta'r"