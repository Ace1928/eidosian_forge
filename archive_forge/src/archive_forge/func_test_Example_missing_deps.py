import pytest
from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.util import to_ternary_int
from spacy.vocab import Vocab
def test_Example_missing_deps():
    vocab = Vocab()
    words = ['I', 'like', 'London', 'and', 'Berlin', '.']
    deps = ['nsubj', 'ROOT', 'dobj', 'cc', 'conj', 'punct']
    heads = [1, 1, 1, 2, 2, 1]
    annots_head_only = {'words': words, 'heads': heads}
    annots_head_dep = {'words': words, 'heads': heads, 'deps': deps}
    predicted = Doc(vocab, words=words)
    example_1 = Example.from_dict(predicted, annots_head_only)
    assert [t.head.i for t in example_1.reference] == [0, 1, 2, 3, 4, 5]
    example_2 = Example.from_dict(predicted, annots_head_dep)
    assert [t.head.i for t in example_2.reference] == heads