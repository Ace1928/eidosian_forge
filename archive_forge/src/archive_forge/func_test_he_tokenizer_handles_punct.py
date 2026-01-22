import pytest
from spacy.lang.he.lex_attrs import like_num
@pytest.mark.parametrize('text,expected_tokens', [('עקבת אחריו בכל רחבי המדינה.', ['עקבת', 'אחריו', 'בכל', 'רחבי', 'המדינה', '.']), ('עקבת אחריו בכל רחבי המדינה?', ['עקבת', 'אחריו', 'בכל', 'רחבי', 'המדינה', '?']), ('עקבת אחריו בכל רחבי המדינה!', ['עקבת', 'אחריו', 'בכל', 'רחבי', 'המדינה', '!']), ('עקבת אחריו בכל רחבי המדינה..', ['עקבת', 'אחריו', 'בכל', 'רחבי', 'המדינה', '..']), ('עקבת אחריו בכל רחבי המדינה...', ['עקבת', 'אחריו', 'בכל', 'רחבי', 'המדינה', '...'])])
def test_he_tokenizer_handles_punct(he_tokenizer, text, expected_tokens):
    tokens = he_tokenizer(text)
    assert expected_tokens == [token.text for token in tokens]