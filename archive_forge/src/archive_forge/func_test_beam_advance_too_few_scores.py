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
def test_beam_advance_too_few_scores(beam, scores):
    n_state = sum((len(beam) for beam in beam))
    scores = scores[:n_state]
    with pytest.raises(IndexError):
        beam.advance(scores[:-1])