import pytest
@pytest.mark.parametrize('text', ['(can)'])
def test_en_tokenizer_splits_no_special(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3