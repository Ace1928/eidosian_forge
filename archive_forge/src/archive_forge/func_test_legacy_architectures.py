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
@pytest.mark.parametrize('name,config', [('entity_linker', {'@architectures': 'spacy.EntityLinker.v1', 'tok2vec': DEFAULT_TOK2VEC_MODEL}), ('entity_linker', {'@architectures': 'spacy.EntityLinker.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL})])
def test_legacy_architectures(name, config):
    vector_length = 3
    nlp = English()
    train_examples = []
    for text, annotation in TRAIN_DATA:
        doc = nlp.make_doc(text)
        train_examples.append(Example.from_dict(doc, annotation))

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        mykb.add_entity(entity='Q7381115', freq=12, entity_vector=[9, 1, -7])
        mykb.add_alias(alias='Russ Cochran', entities=['Q2146908', 'Q7381115'], probabilities=[0.5, 0.5])
        return mykb
    entity_linker = nlp.add_pipe(name, config={'model': config})
    if config['@architectures'] == 'spacy.EntityLinker.v1':
        assert isinstance(entity_linker, EntityLinker_v1)
    else:
        assert isinstance(entity_linker, EntityLinker)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)