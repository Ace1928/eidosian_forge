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
@pytest.mark.issue(4674)
def test_issue4674():
    """Test that setting entities with overlapping identifiers does not mess up IO"""
    nlp = English()
    kb = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
    vector1 = [0.9, 1.1, 1.01]
    vector2 = [1.8, 2.25, 2.01]
    with pytest.warns(UserWarning):
        kb.set_entities(entity_list=['Q1', 'Q1'], freq_list=[32, 111], vector_list=[vector1, vector2])
    assert kb.get_size_entities() == 1
    with make_tempdir() as d:
        dir_path = ensure_path(d)
        if not dir_path.exists():
            dir_path.mkdir()
        file_path = dir_path / 'kb'
        kb.to_disk(str(file_path))
        kb2 = InMemoryLookupKB(nlp.vocab, entity_vector_length=3)
        kb2.from_disk(str(file_path))
    assert kb2.get_size_entities() == 1