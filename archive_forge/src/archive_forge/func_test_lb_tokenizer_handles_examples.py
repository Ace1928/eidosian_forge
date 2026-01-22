import pytest
@pytest.mark.parametrize('text,length', [('»Wat ass mat mir geschitt?«, huet hie geduecht.', 13), ('“Dëst fréi Opstoen”, denkt hien, “mécht ee ganz duercherneen. ', 15), ("Am Grand-Duché ass d'Liewen schéin, mee 't gëtt ze vill Autoen.", 14)])
def test_lb_tokenizer_handles_examples(lb_tokenizer, text, length):
    tokens = lb_tokenizer(text)
    assert len(tokens) == length