import pytest
from spacy import registry
from spacy.pipeline import DependencyParser
from spacy.pipeline._parser_internals.arc_eager import ArcEager
from spacy.pipeline._parser_internals.nonproj import projectivize
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def test_oracle_four_words(arc_eager, vocab):
    words = ['a', 'b', 'c', 'd']
    heads = [1, 1, 3, 3]
    deps = ['left', 'ROOT', 'left', 'ROOT']
    for dep in deps:
        arc_eager.add_action(2, dep)
        arc_eager.add_action(3, dep)
    actions = ['S', 'L-left', 'B-ROOT', 'S', 'D', 'S', 'L-left', 'S', 'D']
    state, cost_history = get_sequence_costs(arc_eager, words, heads, deps, actions)
    expected_gold = [['S'], ['B-ROOT', 'L-left'], ['B-ROOT'], ['S'], ['D'], ['S'], ['L-left'], ['S'], ['D']]
    assert state.is_final()
    for i, state_costs in enumerate(cost_history):
        golds = [act for act, cost in state_costs.items() if cost < 1]
        assert golds == expected_gold[i], (i, golds, expected_gold[i])