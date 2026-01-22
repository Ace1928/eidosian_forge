import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('text,morph,lemma', [('гвоздики', 'Gender=Fem', 'гвоздика'), ('гвоздики', 'Gender=Masc', 'гвоздик'), ('вина', 'Gender=Fem', 'вина'), ('вина', 'Gender=Neut', 'вино')])
def test_ru_lemmatizer_works_with_noun_homonyms(ru_lemmatizer, text, morph, lemma):
    doc = Doc(ru_lemmatizer.vocab, words=[text], pos=['NOUN'], morphs=[morph])
    result_lemmas = ru_lemmatizer.pymorphy2_lemmatize(doc[0])
    assert result_lemmas == [lemma]