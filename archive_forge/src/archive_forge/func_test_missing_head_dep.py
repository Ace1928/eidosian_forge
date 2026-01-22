import numpy
import pytest
from spacy.attrs import IS_ALPHA, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_STOP, IS_TITLE
from spacy.symbols import VERB
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_missing_head_dep(en_vocab):
    """Check that the Doc constructor and Example.from_dict parse missing information the same"""
    heads = [1, 1, 1, 1, 2, None]
    deps = ['', 'ROOT', 'dobj', 'cc', 'conj', None]
    words = ['I', 'like', 'London', 'and', 'Berlin', '.']
    doc = Doc(en_vocab, words=words, heads=heads, deps=deps)
    pred_has_heads = [t.has_head() for t in doc]
    pred_has_deps = [t.has_dep() for t in doc]
    pred_heads = [t.head.i for t in doc]
    pred_deps = [t.dep_ for t in doc]
    pred_sent_starts = [t.is_sent_start for t in doc]
    assert pred_has_heads == [False, True, True, True, True, False]
    assert pred_has_deps == [False, True, True, True, True, False]
    assert pred_heads[1:5] == [1, 1, 1, 2]
    assert pred_deps[1:5] == ['ROOT', 'dobj', 'cc', 'conj']
    assert pred_sent_starts == [True, False, False, False, False, False]
    example = Example.from_dict(doc, {'heads': heads, 'deps': deps})
    ref_has_heads = [t.has_head() for t in example.reference]
    ref_has_deps = [t.has_dep() for t in example.reference]
    ref_heads = [t.head.i for t in example.reference]
    ref_deps = [t.dep_ for t in example.reference]
    ref_sent_starts = [t.is_sent_start for t in example.reference]
    assert ref_has_heads == pred_has_heads
    assert ref_has_deps == pred_has_heads
    assert ref_heads == pred_heads
    assert ref_deps == pred_deps
    assert ref_sent_starts == pred_sent_starts
    aligned_heads, aligned_deps = example.get_aligned_parse(projectivize=True)
    assert aligned_deps[0] == ref_deps[0]
    assert aligned_heads[0] == ref_heads[0]
    assert aligned_deps[5] == ref_deps[5]
    assert aligned_heads[5] == ref_heads[5]