import pytest
@pytest.mark.parametrize('text', ['Donaudampfschifffahrtsgesellschaftskapitänsanwärterposten', 'Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz', 'Kraftfahrzeug-Haftpflichtversicherung', 'Vakuum-Mittelfrequenz-Induktionsofen'])
def test_de_tokenizer_handles_long_words(de_tokenizer, text):
    tokens = de_tokenizer(text)
    assert len(tokens) == 1