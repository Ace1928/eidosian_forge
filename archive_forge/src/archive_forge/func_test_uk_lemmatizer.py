import pytest
from spacy.tokens import Doc
def test_uk_lemmatizer(uk_lemmatizer):
    """Check that the default uk lemmatizer runs."""
    doc = Doc(uk_lemmatizer.vocab, words=['a', 'b', 'c'])
    assert uk_lemmatizer.mode == 'pymorphy3'
    uk_lemmatizer(doc)
    assert [token.lemma for token in doc]