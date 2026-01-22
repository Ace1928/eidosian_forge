import pytest
def test_es_tokenizer_handles_exc_in_text(es_tokenizer):
    text = 'Mariano Rajoy ha corrido aprox. medio kilómetro'
    tokens = es_tokenizer(text)
    assert len(tokens) == 7
    assert tokens[4].text == 'aprox.'