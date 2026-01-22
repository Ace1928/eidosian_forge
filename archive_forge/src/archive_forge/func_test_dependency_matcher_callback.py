import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_dependency_matcher_callback(en_vocab, doc):
    pattern = [{'RIGHT_ID': 'quick', 'RIGHT_ATTRS': {'ORTH': 'quick'}}]
    nomatch_pattern = [{'RIGHT_ID': 'quick', 'RIGHT_ATTRS': {'ORTH': 'NOMATCH'}}]
    matcher = DependencyMatcher(en_vocab)
    mock = Mock()
    matcher.add('pattern', [pattern], on_match=mock)
    matcher.add('nomatch_pattern', [nomatch_pattern], on_match=mock)
    matches = matcher(doc)
    assert len(matches) == 1
    mock.assert_called_once_with(matcher, doc, 0, matches)
    matcher2 = DependencyMatcher(en_vocab)
    matcher2.add('pattern', [pattern])
    matches2 = matcher2(doc)
    assert matches == matches2