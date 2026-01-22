import pytest
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.tokens.doc import Doc
from spacy.vocab import Vocab
def test_push_pop(doc):
    state = StateClass(doc)
    state.push()
    assert state.buffer_length() == 3
    assert state.stack == [0]
    assert 0 not in state.queue
    state.push()
    assert state.stack == [1, 0]
    assert 1 not in state.queue
    assert state.buffer_length() == 2
    state.pop()
    assert state.stack == [0]
    assert 1 not in state.queue