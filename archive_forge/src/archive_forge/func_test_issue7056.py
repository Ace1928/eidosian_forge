import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
@pytest.mark.issue(7056)
def test_issue7056():
    """Test that the Unshift transition works properly, and doesn't cause
    sentence segmentation errors."""
    vocab = Vocab()
    ae = ArcEager(vocab.strings, ArcEager.get_actions(left_labels=['amod'], right_labels=['pobj']))
    doc = Doc(vocab, words='Severe pain , after trauma'.split())
    state = ae.init_batch([doc])[0]
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'L-amod')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'S')
    ae.apply_transition(state, 'R-pobj')
    ae.apply_transition(state, 'D')
    ae.apply_transition(state, 'D')
    ae.apply_transition(state, 'D')
    assert not state.eol()