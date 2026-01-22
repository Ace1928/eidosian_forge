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
@pytest.mark.issue(9137)
def test_kb_serialize_2(nlp):
    v = [5, 6, 7, 8]
    kb1 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
    kb1.set_entities(['E1'], [1], [v])
    assert kb1.get_vector('E1') == v
    with make_tempdir() as d:
        kb1.to_disk(d / 'kb')
        kb2 = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=4)
        kb2.from_disk(d / 'kb')
        assert kb2.get_vector('E1') == v