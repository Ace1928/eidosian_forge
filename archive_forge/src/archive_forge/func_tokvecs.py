import hypothesis
import hypothesis.strategies
import numpy
import pytest
from thinc.tests.strategies import ndarrays_of_shape
from spacy.language import Language
from spacy.pipeline._parser_internals._beam_utils import BeamBatch
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.fixture
def tokvecs(docs, vector_size):
    output = []
    for doc in docs:
        vec = numpy.random.uniform(-0.1, 0.1, (len(doc), vector_size))
        output.append(numpy.asarray(vec))
    return output