import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def test_invalid_greediness(doc, text):
    matcher = Matcher(doc.vocab)
    with pytest.raises(ValueError):
        matcher.add('RULE', [pattern1], greedy='GREEDY')