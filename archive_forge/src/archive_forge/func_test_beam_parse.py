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
def test_beam_parse(examples, beam_width):
    nlp = Language()
    parser = nlp.add_pipe('beam_parser')
    parser.cfg['beam_width'] = beam_width
    parser.add_label('nsubj')
    parser.initialize(lambda: examples)
    doc = nlp.make_doc('Australia is a country')
    parser(doc)