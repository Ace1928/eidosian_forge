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
@pytest.mark.issue(9575)
def test_tokenization_mismatch():
    nlp = English()
    doc1 = Doc(nlp.vocab, words=['Kirby', '123456'], spaces=[True, False], ents=['B-CHARACTER', 'B-CARDINAL'])
    doc2 = Doc(nlp.vocab, words=['Kirby', '123', '456'], spaces=[True, False, False], ents=['B-CHARACTER', 'B-CARDINAL', 'B-CARDINAL'])
    eg = Example(doc1, doc2)
    train_examples = [eg]
    vector_length = 3

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q613241', freq=12, entity_vector=[6, -4, 3])
        mykb.add_alias('Kirby', ['Q613241'], [0.9])
        return mykb
    entity_linker = nlp.add_pipe('entity_linker', last=True)
    entity_linker.set_kb(create_kb)
    optimizer = nlp.initialize(get_examples=lambda: train_examples)
    for i in range(2):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)
    nlp.add_pipe('sentencizer', first=True)
    nlp.evaluate(train_examples)