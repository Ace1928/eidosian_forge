import pytest
from spacy.lang.es import Spanish
from spacy.lang.es.lex_attrs import like_num
@pytest.mark.parametrize('text,length', [('¿Por qué José Mujica?', 6), ('“¿Oh no?”', 6), ('¡Sí! "Vámonos", contestó José Arcadio Buendía', 11), ('Corrieron aprox. 10km.', 5), ('Y entonces por qué...', 5)])
def test_es_tokenizer_handles_cnts(es_tokenizer, text, length):
    tokens = es_tokenizer(text)
    assert len(tokens) == length