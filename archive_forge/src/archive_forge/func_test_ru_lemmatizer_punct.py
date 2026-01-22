import pytest
from spacy.tokens import Doc
def test_ru_lemmatizer_punct(ru_lemmatizer):
    doc = Doc(ru_lemmatizer.vocab, words=['«'], pos=['PUNCT'])
    assert ru_lemmatizer.pymorphy2_lemmatize(doc[0]) == ['"']
    doc = Doc(ru_lemmatizer.vocab, words=['»'], pos=['PUNCT'])
    assert ru_lemmatizer.pymorphy2_lemmatize(doc[0]) == ['"']