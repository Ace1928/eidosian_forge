import pytest
@pytest.mark.parametrize('text', ['best-known'])
def test_en_tokenizer_splits_hyphens(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3