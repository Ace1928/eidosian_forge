import pytest
@pytest.mark.parametrize('text', ['Jul.', 'jul.', 'sön.', 'Sön.'])
def test_sv_tokenizer_handles_ambiguous_abbr(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 2