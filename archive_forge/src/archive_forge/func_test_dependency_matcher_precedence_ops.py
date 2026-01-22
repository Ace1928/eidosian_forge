import copy
import pickle
import re
import pytest
from mock import Mock
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from ..doc.test_underscore import clean_underscore  # noqa: F401
@pytest.mark.parametrize('op,num_matches', [('.', 8), ('.*', 20), (';', 8), (';*', 20)])
def test_dependency_matcher_precedence_ops(en_vocab, op, num_matches):
    doc = Doc(en_vocab, words=['a', 'b', 'c', 'd', 'e'] * 2, heads=[0, 0, 0, 0, 0, 5, 5, 5, 5, 5], deps=['dep'] * 10)
    match_count = 0
    for text in ['a', 'b', 'c', 'd', 'e']:
        pattern = [{'RIGHT_ID': '1', 'RIGHT_ATTRS': {'ORTH': text}}, {'LEFT_ID': '1', 'REL_OP': op, 'RIGHT_ID': '2', 'RIGHT_ATTRS': {}}]
        matcher = DependencyMatcher(en_vocab)
        matcher.add('A', [pattern])
        matches = matcher(doc)
        match_count += len(matches)
        for match in matches:
            match_id, token_ids = match
            if op == '.':
                assert token_ids[0] == token_ids[1] - 1
            elif op == ';':
                assert token_ids[0] == token_ids[1] + 1
            elif op == '.*':
                assert token_ids[0] < token_ids[1]
            elif op == ';*':
                assert token_ids[0] > token_ids[1]
            assert doc[token_ids[0]].sent == doc[token_ids[1]].sent
    assert match_count == num_matches