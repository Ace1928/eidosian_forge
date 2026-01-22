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
def test_issue_1971_3(en_vocab):
    """Test that pattern matches correctly for multiple extension attributes."""
    Token.set_extension('a', default=1, force=True)
    Token.set_extension('b', default=2, force=True)
    doc = Doc(en_vocab, words=['hello', 'world'])
    matcher = Matcher(en_vocab)
    matcher.add('A', [[{'_': {'a': 1}}]])
    matcher.add('B', [[{'_': {'b': 2}}]])
    matches = sorted(((en_vocab.strings[m_id], s, e) for m_id, s, e in matcher(doc)))
    assert len(matches) == 4
    assert matches == sorted([('A', 0, 1), ('A', 1, 2), ('B', 0, 1), ('B', 1, 2)])