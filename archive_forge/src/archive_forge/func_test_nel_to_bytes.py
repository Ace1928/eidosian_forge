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
def test_nel_to_bytes():

    def create_kb(vocab):
        kb = InMemoryLookupKB(vocab, entity_vector_length=3)
        kb.add_entity(entity='Q2146908', freq=12, entity_vector=[6, -4, 3])
        kb.add_alias(alias='Russ Cochran', entities=['Q2146908'], probabilities=[0.8])
        return kb
    nlp_1 = English()
    nlp_1.add_pipe('ner')
    entity_linker_1 = nlp_1.add_pipe('entity_linker', last=True)
    entity_linker_1.set_kb(create_kb)
    assert entity_linker_1.kb.contains_alias('Russ Cochran')
    assert nlp_1.pipe_names == ['ner', 'entity_linker']
    nlp_bytes = nlp_1.to_bytes()
    nlp_2 = English()
    nlp_2.add_pipe('ner')
    nlp_2.add_pipe('entity_linker', last=True)
    assert nlp_2.pipe_names == ['ner', 'entity_linker']
    assert not nlp_2.get_pipe('entity_linker').kb.contains_alias('Russ Cochran')
    nlp_2 = nlp_2.from_bytes(nlp_bytes)
    kb_2 = nlp_2.get_pipe('entity_linker').kb
    assert kb_2.contains_alias('Russ Cochran')
    assert kb_2.get_vector('Q2146908') == [6, -4, 3]
    assert_almost_equal(kb_2.get_prior_prob(entity='Q2146908', alias='Russ Cochran'), 0.8)