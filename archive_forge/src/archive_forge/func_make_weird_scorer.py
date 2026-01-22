import numpy
import pytest
from spacy import registry, util
from spacy.lang.en import English
from spacy.pipeline import AttributeRuler
from spacy.tokens import Doc
from spacy.training import Example
from ..util import make_tempdir
@registry.misc('weird_scorer.v1')
def make_weird_scorer():

    def weird_scorer(examples, weird_score, **kwargs):
        return {'weird_score': weird_score}
    return weird_scorer