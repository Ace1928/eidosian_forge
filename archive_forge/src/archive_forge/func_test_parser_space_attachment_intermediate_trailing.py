import pytest
from spacy.tokens import Doc
from ..util import apply_transition_sequence
@pytest.mark.skip(reason='The step_through API was removed (but should be brought back)')
def test_parser_space_attachment_intermediate_trailing(en_vocab, en_parser):
    words = ['This', 'is', '\t', 'a', '\t\n', '\n', 'sentence', '.', '\n\n', '\n']
    heads = [1, 1, 1, 5, 3, 1, 1, 6]
    transition = ['L-nsubj', 'S', 'L-det', 'R-attr', 'D', 'R-punct']
    doc = Doc(en_vocab, words=words, heads=heads)
    assert doc[2].is_space
    assert doc[4].is_space
    assert doc[5].is_space
    assert doc[8].is_space
    assert doc[9].is_space
    apply_transition_sequence(en_parser, doc, transition)
    for token in doc:
        assert token.dep != 0 or token.is_space
    assert [token.head.i for token in doc] == [1, 1, 1, 6, 3, 3, 1, 1, 7, 7]