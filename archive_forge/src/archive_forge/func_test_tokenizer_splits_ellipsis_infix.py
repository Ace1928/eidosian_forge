import pytest
@pytest.mark.parametrize('text', ['svart...Gul', 'svart...gul'])
def test_tokenizer_splits_ellipsis_infix(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 3