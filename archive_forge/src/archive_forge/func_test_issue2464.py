import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(2464)
def test_issue2464(en_vocab):
    """Test problem with successive ?. This is the same bug, so putting it here."""
    matcher = Matcher(en_vocab)
    doc = Doc(en_vocab, words=['a', 'b'])
    matcher.add('4', [[{'OP': '?'}, {'OP': '?'}]])
    matches = matcher(doc)
    assert len(matches) == 3