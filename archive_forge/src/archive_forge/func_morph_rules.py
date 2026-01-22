import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
@pytest.fixture
def morph_rules():
    return {'DT': {'the': {'POS': 'DET', 'LEMMA': 'a', 'Case': 'Nom'}}}