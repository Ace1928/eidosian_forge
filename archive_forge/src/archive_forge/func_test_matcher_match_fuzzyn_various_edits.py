import pytest
from mock import Mock
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('fuzzyn', range(1, 6))
def test_matcher_match_fuzzyn_various_edits(en_vocab, fuzzyn):
    matcher = Matcher(en_vocab)
    matcher.add('GoogleNow', [[{'ORTH': {f'FUZZY{fuzzyn}': 'GoogleNow'}}]])
    words = ['GoogleNow', 'GoogleNuw', 'GoogleNuew', 'GoogleNoweee', 'GiggleNuw3', 'gouggle5New']
    doc = Doc(en_vocab, words)
    assert len(matcher(doc)) == fuzzyn + 1