import pickle
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.lookups import Lookups
from ..util import make_tempdir
def test_lemmatizer_config(nlp):
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'rule'})
    nlp.initialize()
    doc = nlp.make_doc('coping')
    with pytest.warns(UserWarning):
        doc = lemmatizer(doc)
    doc = lemmatizer(doc)
    doc = nlp.make_doc('coping')
    assert doc[0].lemma_ == ''
    doc[0].pos_ = 'VERB'
    doc = lemmatizer(doc)
    doc = lemmatizer(doc)
    assert doc[0].text == 'coping'
    assert doc[0].lemma_ == 'cope'
    doc = nlp.make_doc('coping')
    doc[0].pos_ = 'VERB'
    assert doc[0].lemma_ == ''
    doc = lemmatizer(doc)
    assert doc[0].text == 'coping'
    assert doc[0].lemma_ == 'cope'