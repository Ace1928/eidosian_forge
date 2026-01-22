import pytest
@pytest.mark.parametrize('text', ["aujourd'hui", "Aujourd'hui", "prud'hommes", 'prud’hommal', 'audio-numérique', 'Audio-numérique', "entr'amis", "entr'abat", "rentr'ouvertes", "grand'hamien", 'Châteauneuf-la-Forêt', 'Château-Guibert', 'refox-trottâmes', "z'yeutes", 'black-outeront', 'états-unienne', 'courtes-pattes', 'court-pattes', 'saut-de-ski', 'Écourt-Saint-Quentin', "Bout-de-l'Îlien", "pet-en-l'air"])
def test_fr_tokenizer_infix_exceptions(fr_tokenizer, text):
    tokens = fr_tokenizer(text)
    assert len(tokens) == 1