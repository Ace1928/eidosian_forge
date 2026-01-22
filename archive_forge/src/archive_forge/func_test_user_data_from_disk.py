from spacy.compat import pickle
from spacy.language import Language
def test_user_data_from_disk():
    nlp = Language()
    doc = nlp('Hello')
    doc.user_data[0, 1] = False
    b = doc.to_bytes()
    doc2 = doc.__class__(doc.vocab).from_bytes(b)
    assert doc2.user_data[0, 1] is False