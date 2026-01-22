import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(590)
def test_issue590(en_vocab):
    """Test overlapping matches"""
    doc = Doc(en_vocab, words=['n', '=', '1', ';', 'a', ':', '5', '%'])
    matcher = Matcher(en_vocab)
    matcher.add('ab', [[{'IS_ALPHA': True}, {'ORTH': ':'}, {'LIKE_NUM': True}, {'ORTH': '%'}]])
    matcher.add('ab', [[{'IS_ALPHA': True}, {'ORTH': '='}, {'LIKE_NUM': True}]])
    matches = matcher(doc)
    assert len(matches) == 2