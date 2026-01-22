import numpy
import pytest
from spacy.attrs import DEP, MORPH, ORTH, POS, SHAPE
from spacy.tokens import Doc
@pytest.mark.issue(2203)
def test_issue2203(en_vocab):
    """Test that lemmas are set correctly in doc.from_array."""
    words = ['I', "'ll", 'survive']
    tags = ['PRP', 'MD', 'VB']
    lemmas = ['-PRON-', 'will', 'survive']
    tag_ids = [en_vocab.strings.add(tag) for tag in tags]
    lemma_ids = [en_vocab.strings.add(lemma) for lemma in lemmas]
    doc = Doc(en_vocab, words=words)
    doc.from_array('TAG', numpy.array(tag_ids, dtype='uint64'))
    doc.from_array('LEMMA', numpy.array(lemma_ids, dtype='uint64'))
    assert [t.tag_ for t in doc] == tags
    assert [t.lemma_ for t in doc] == lemmas
    doc_array = doc.to_array(['TAG', 'LEMMA'])
    new_doc = Doc(doc.vocab, words=words).from_array(['TAG', 'LEMMA'], doc_array)
    assert [t.tag_ for t in new_doc] == tags
    assert [t.lemma_ for t in new_doc] == lemmas