import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(3328)
def test_issue3328(en_vocab):
    doc = Doc(en_vocab, words=['Hello', ',', 'how', 'are', 'you', 'doing', '?'])
    matcher = Matcher(en_vocab)
    patterns = [[{'LOWER': {'IN': ['hello', 'how']}}], [{'LOWER': {'IN': ['you', 'doing']}}]]
    matcher.add('TEST', patterns)
    matches = matcher(doc)
    assert len(matches) == 4
    matched_texts = [doc[start:end].text for _, start, end in matches]
    assert matched_texts == ['Hello', 'how', 'you', 'doing']