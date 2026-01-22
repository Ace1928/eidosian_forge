import pytest
@pytest.mark.parametrize('text', ['bl.a', 'm.a.o.', 'Jan.', 'Dec.', 'kr.', 'osv.'])
def test_sv_tokenizer_handles_abbr(sv_tokenizer, text):
    tokens = sv_tokenizer(text)
    assert len(tokens) == 1