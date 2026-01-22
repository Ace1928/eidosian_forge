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
def test_label_types():
    nlp = Language()
    morphologizer = nlp.add_pipe('morphologizer')
    morphologizer.add_label('Feat=A')
    with pytest.raises(ValueError):
        morphologizer.add_label(9)