import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('word,lemma', (('якийсь', 'якийсь'), ('розповідають', 'розповідати'), ('розповіси', 'розповісти')))
def test_uk_lookup_lemmatizer(uk_lookup_lemmatizer, word, lemma):
    assert uk_lookup_lemmatizer.mode == 'pymorphy3_lookup'
    doc = Doc(uk_lookup_lemmatizer.vocab, words=[word])
    assert uk_lookup_lemmatizer(doc)[0].lemma_ == lemma