import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
@pytest.mark.issue(242)
def test_issue242(en_tokenizer):
    """Test overlapping multi-word phrases."""
    text = 'There are different food safety standards in different countries.'
    patterns = [[{'LOWER': 'food'}, {'LOWER': 'safety'}], [{'LOWER': 'safety'}, {'LOWER': 'standards'}]]
    doc = en_tokenizer(text)
    matcher = Matcher(doc.vocab)
    matcher.add('FOOD', patterns)
    matches = [(ent_type, start, end) for ent_type, start, end in matcher(doc)]
    match1, match2 = matches
    assert match1[1] == 3
    assert match1[2] == 5
    assert match2[1] == 4
    assert match2[2] == 6
    with pytest.raises(ValueError):
        doc.ents += tuple(matches)