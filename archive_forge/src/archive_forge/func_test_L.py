import pytest
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
def test_L(doc):
    state = StateClass(doc)
    assert state.L(2, 1) == -1
    state.add_arc(2, 1, 0)
    assert state.arcs == [{'head': 2, 'child': 1, 'label': 0}]
    assert state.L(2, 1) == 1
    state.add_arc(2, 0, 0)
    assert state.L(2, 1) == 0
    assert state.n_L(2) == 2