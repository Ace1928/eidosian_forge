import pytest
from spacy.tokens import Doc
from ..util import apply_transition_sequence
@pytest.mark.skip(reason='The step_through API was removed (but should be brought back)')
def test_parser_space_attachment_leading(en_vocab, en_parser):
    words = ['\t', '\n', 'This', 'is', 'a', 'sentence', '.']
    heads = [1, 2, 2, 4, 2, 2]
    doc = Doc(en_vocab, words=words, heads=heads)
    assert doc[0].is_space
    assert doc[1].is_space
    assert doc[2].text == 'This'
    with en_parser.step_through(doc) as stepwise:
        pass
    assert doc[0].head.i == 2
    assert doc[1].head.i == 2
    assert stepwise.stack == set([2])