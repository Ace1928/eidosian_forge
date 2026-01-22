import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.usefixtures('clean_underscore')
def test_dependency_matcher_span_user_data(en_tokenizer):
    doc = en_tokenizer('a b c d e')
    for token in doc:
        token.head = doc[0]
        token.dep_ = 'a'
    Token.set_extension('is_c', default=False)
    doc[2]._.is_c = True
    pattern = [{'RIGHT_ID': 'c', 'RIGHT_ATTRS': {'_': {'is_c': True}}}]
    matcher = DependencyMatcher(en_tokenizer.vocab)
    matcher.add('C', [pattern])
    doc_matches = matcher(doc)
    offset = 1
    span_matches = matcher(doc[offset:])
    for doc_match, span_match in zip(sorted(doc_matches), sorted(span_matches)):
        assert doc_match[0] == span_match[0]
        for doc_t_i, span_t_i in zip(doc_match[1], span_match[1]):
            assert doc_t_i == span_t_i + offset