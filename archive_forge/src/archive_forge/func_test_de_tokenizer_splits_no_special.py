import pytest
@pytest.mark.parametrize('text', ['(unter)'])
def test_de_tokenizer_splits_no_special(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 3