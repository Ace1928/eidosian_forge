import copy
import pickle
import numpy
import pytest
from spacy.attrs import DEP, HEAD
from spacy.lang.en import English
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
@pytest.mark.issue(1727)
def test_issue1727():
    """Test that models with no pretrained vectors can be deserialized
    correctly after vectors are added."""
    nlp = Language(Vocab())
    data = numpy.ones((3, 300), dtype='f')
    vectors = Vectors(data=data, keys=['I', 'am', 'Matt'])
    tagger = nlp.create_pipe('tagger')
    tagger.add_label('PRP')
    assert tagger.cfg.get('pretrained_dims', 0) == 0
    tagger.vocab.vectors = vectors
    with make_tempdir() as path:
        tagger.to_disk(path)
        tagger = nlp.create_pipe('tagger').from_disk(path)
        assert tagger.cfg.get('pretrained_dims', 0) == 0