import re
import pytest
from spacy.attrs import IS_PUNCT, LOWER, ORTH
from spacy.errors import MatchPatternError
from spacy.lang.en import English
from spacy.lang.lex_attrs import LEX_ATTRS
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
def merge_phrases(matcher, doc, i, matches):
    """Merge a phrase. We have to be careful here because we'll change the
        token indices. To avoid problems, merge all the phrases once we're called
        on the last match."""
    if i != len(matches) - 1:
        return None
    spans = [Span(doc, start, end, label=label) for label, start, end in matches]
    with doc.retokenize() as retokenizer:
        for span in spans:
            tag = 'NNP' if span.label_ else span.root.tag_
            attrs = {'tag': tag, 'lemma': span.text}
            retokenizer.merge(span, attrs=attrs)
            doc.ents = doc.ents + (span,)