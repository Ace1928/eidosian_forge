import pytest
from numpy.testing import assert_almost_equal, assert_equal
from thinc.api import get_current_ops
from spacy import util
from spacy.attrs import MORPH
from spacy.lang.en import English
from spacy.language import Language
from spacy.morphology import Morphology
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc
from spacy.training import Example
def test_implicit_label():
    nlp = Language()
    nlp.add_pipe('morphologizer')
    train_examples = []
    for t in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    nlp.initialize(get_examples=lambda: train_examples)