import pytest
def test_lt_tokenizer_handles_long_text(lt_tokenizer):
    text = 'Tokios sausros kriterijus atitinka pirmadienį atlikti skaičiavimai, palyginus faktinį ir žemiausią vidutinį daugiametį vandens lygį. Nustatyta, kad iš 48 šalies vandens matavimo stočių 28-iose stotyse vandens lygis yra žemesnis arba lygus žemiausiam vidutiniam daugiamečiam šiltojo laikotarpio vandens lygiui.'
    tokens = lt_tokenizer(text)
    assert len(tokens) == 42