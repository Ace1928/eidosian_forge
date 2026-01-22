import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(3549)
def test_issue3549(en_vocab):
    """Test that match pattern validation doesn't raise on empty errors."""
    matcher = Matcher(en_vocab, validate=True)
    pattern = [{'LOWER': 'hello'}, {'LOWER': 'world'}]
    matcher.add('GOOD', [pattern])
    with pytest.raises(MatchPatternError):
        matcher.add('BAD', [[{'X': 'Y'}]])