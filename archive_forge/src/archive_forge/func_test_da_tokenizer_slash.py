import pytest
@pytest.mark.parametrize('text,n_tokens', [('Godt og/eller skidt', 3), ('Kør 4 km/t på vejen', 5), ('Det blæser 12 m/s.', 5), ('Det blæser 12 m/sek. på havnen', 6), ('Windows 8/Windows 10', 5), ('Billeten virker til bus/tog/metro', 8), ('26/02/2019', 1), ('Kristiansen c/o Madsen', 3), ('Sprogteknologi a/s', 2), ('De boede i A/B Bellevue', 5), ('Jeg købte billet t/r.', 5), ('Murerarbejdsmand m/k søges', 3), ('Netværket kører over TCP/IP', 4)])
def test_da_tokenizer_slash(da_tokenizer, text, n_tokens):
    tokens = da_tokenizer(text)
    assert len(tokens) == n_tokens