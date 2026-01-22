import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.parametrize('pattern,re_pattern', [(pattern1, re_pattern1), (pattern2, re_pattern2), (pattern3, re_pattern3), (pattern4, re_pattern4), (pattern5, re_pattern5)])
def test_match_consuming(doc, text, pattern, re_pattern):
    """Test that matcher.__call__ consumes tokens on a match similar to
    re.findall."""
    matcher = Matcher(doc.vocab)
    matcher.add(re_pattern, [pattern], greedy='FIRST')
    matches = matcher(doc)
    re_matches = [m.span() for m in re.finditer(re_pattern, text)]
    assert len(matches) == len(re_matches)