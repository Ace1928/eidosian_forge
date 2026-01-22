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
@pytest.mark.filterwarnings('ignore:\\[W036')
def test_append_invalid_alias(nlp):
    """Test that append an alias will throw an error if prior probs are exceeding 1"""
    mykb = InMemoryLookupKB(nlp.vocab, entity_vector_length=1)
    mykb.add_entity(entity='Q1', freq=27, entity_vector=[1])
    mykb.add_entity(entity='Q2', freq=12, entity_vector=[2])
    mykb.add_entity(entity='Q3', freq=5, entity_vector=[3])
    mykb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.1])
    mykb.add_alias(alias='adam', entities=['Q2'], probabilities=[0.9])
    with pytest.raises(ValueError):
        mykb.append_alias(alias='douglas', entity='Q1', prior_prob=0.2)