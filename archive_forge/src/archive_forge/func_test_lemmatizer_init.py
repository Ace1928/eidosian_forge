import pickle
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.lookups import Lookups
from ..util import make_tempdir
def test_lemmatizer_init(nlp):
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'lookup'})
    assert isinstance(lemmatizer.lookups, Lookups)
    assert not lemmatizer.lookups.tables
    assert lemmatizer.mode == 'lookup'
    with pytest.raises(ValueError):
        nlp('test')
    nlp.initialize()
    assert lemmatizer.lookups.tables
    assert nlp('cope')[0].lemma_ == 'cope'
    assert nlp('coped')[0].lemma_ == 'cope'
    lemmatizer.lookups = Lookups()
    assert nlp('cope')[0].lemma_ == 'cope'
    assert nlp('coped')[0].lemma_ == 'coped'
    nlp.remove_pipe('lemmatizer')
    lemmatizer = nlp.add_pipe('lemmatizer', config={'mode': 'lookup'})
    with pytest.raises(ValueError):
        lemmatizer.initialize(lookups=Lookups())
    lookups = Lookups()
    lookups.add_table('lemma_lookup', {})
    lemmatizer.initialize(lookups=lookups)