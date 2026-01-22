import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.filterwarnings('ignore:\\[W036')
def test_matcher_remove():
    nlp = English()
    matcher = Matcher(nlp.vocab)
    text = 'This is a test case.'
    pattern = [{'ORTH': 'test'}, {'OP': '?'}]
    assert len(matcher) == 0
    matcher.add('Rule', [pattern])
    assert 'Rule' in matcher
    results1 = matcher(nlp(text))
    assert len(results1) == 2
    matcher.remove('Rule')
    results2 = matcher(nlp(text))
    assert len(results2) == 0
    with pytest.raises(ValueError):
        matcher.remove('Rule')