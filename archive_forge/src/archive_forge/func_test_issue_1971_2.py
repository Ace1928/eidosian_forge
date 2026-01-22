import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(1971)
def test_issue_1971_2(en_vocab):
    matcher = Matcher(en_vocab)
    pattern1 = [{'ORTH': 'EUR', 'LOWER': {'IN': ['eur']}}, {'LIKE_NUM': True}]
    pattern2 = [{'LIKE_NUM': True}, {'ORTH': 'EUR'}]
    doc = Doc(en_vocab, words=['EUR', '10', 'is', '10', 'EUR'])
    matcher.add('TEST1', [pattern1, pattern2])
    matches = matcher(doc)
    assert len(matches) == 2