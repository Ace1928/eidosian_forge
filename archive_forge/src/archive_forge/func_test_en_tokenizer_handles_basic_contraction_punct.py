import pytest
@pytest.mark.parametrize('text', ["`ain't", '"isn\'t', "can't!"])
def test_en_tokenizer_handles_basic_contraction_punct(en_tokenizer, text):
    tokens = en_tokenizer(text)
    assert len(tokens) == 3