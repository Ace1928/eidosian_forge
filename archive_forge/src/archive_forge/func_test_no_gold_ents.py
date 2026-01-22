from typing import Any, Callable, Dict, Iterable, Tuple
import pytest
from numpy.testing import assert_equal
from spacy import Language, registry, util
from spacy.attrs import ENT_KB_ID
from spacy.compat import pickle
from spacy.kb import Candidate, InMemoryLookupKB, KnowledgeBase, get_candidates
from spacy.lang.en import English
from spacy.ml import load_kb
from spacy.ml.models.entity_linker import build_span_maker
from spacy.pipeline import EntityLinker
from spacy.pipeline.legacy import EntityLinker_v1
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.scorer import Scorer
from spacy.tests.util import make_tempdir
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.util import ensure_path
from spacy.vocab import Vocab
@pytest.mark.parametrize('patterns', [[{'label': 'CHARACTER', 'pattern': 'Kirby'}], [{'label': 'PERSON', 'pattern': 'Korby'}], [{'label': 'IS', 'pattern': 'is'}, {'label': 'COLOR', 'pattern': 'pink'}]])
def test_no_gold_ents(patterns):
    TRAIN_DATA = [('Kirby is pink', {'links': {(0, 5): {'Q613241': 1.0}}, 'entities': [(0, 5, 'CHARACTER')], 'sent_starts': [1, 0, 0]})]
    nlp = English()
    vector_length = 3
    train_examples = []
    for text, annotation in TRAIN_DATA:
        doc = nlp(text)
        train_examples.append(Example.from_dict(doc, annotation))
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    for eg in train_examples:
        eg.predicted = ruler(eg.predicted)
    nlp.remove_pipe('entity_ruler')

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q613241', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Kirby', ['Q613241'], [0.9])
        mykb.add_entity(entity='pink', freq=12, entity_vector=[7, 2, -5])
        mykb.add_alias('pink', ['pink'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', config={'use_gold_ents': False}, last=True)
    entity_linker.set_kb(create_kb)
    assert entity_linker.use_gold_ents is False
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    nlp.evaluate(train_examples)