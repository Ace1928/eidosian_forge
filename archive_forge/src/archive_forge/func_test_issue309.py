import pytest
from spacy.tokens import Doc
from ...util import apply_transition_sequence
@pytest.mark.issue(309)
def test_issue309(en_vocab):
    """Test Issue #309: SBD fails on empty string"""
    doc = Doc(en_vocab, words=[' '], heads=[0], deps=['ROOT'])
    assert len(doc) == 1
    sents = list(doc.sents)
    assert len(sents) == 1