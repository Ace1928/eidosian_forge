import pytest
@pytest.mark.issue(12311)
@pytest.mark.parametrize('text', ['99:e', 'c:a', 'EU:s', 'Maj:t'])
def test_sv_tokenizer_handles_colon(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 1