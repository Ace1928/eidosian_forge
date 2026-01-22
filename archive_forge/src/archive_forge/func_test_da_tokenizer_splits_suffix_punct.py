import pytest
@pytest.mark.parametrize('text', ["ta'r)"])
def test_da_tokenizer_splits_suffix_punct(da_tokenizer, text):
    tokens = da_tokenizer(text)
    assert len(tokens) == 2
    assert tokens[0].text == "ta'r"
    assert tokens[1].text == ')'