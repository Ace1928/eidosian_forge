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
def test_el_pipe_configuration(nlp):
    """Test correct candidate generation as part of the EL pipe"""
    nlp.add_pipe('sentencizer')
    pattern = {'label': 'PERSON', 'pattern': [{'LOWER': 'douglas'}]}
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns([pattern])

    def create_kb(vocab):
        kb = InMemoryLookupKB(vocab, entity_vector_length=1)
        kb.add_entity(entity='Q2', freq=12, entity_vector=[2])
        kb.add_entity(entity='Q3', freq=5, entity_vector=[3])
        kb.add_alias(alias='douglas', entities=['Q2', 'Q3'], probabilities=[0.8, 0.1])
        return kb
    entity_linker = nlp.add_pipe('entity_linker', config={'incl_context': False})
    entity_linker.set_kb(create_kb)
    text = 'Douglas and douglas are not the same.'
    doc = nlp(text)
    assert doc[0].ent_kb_id_ == 'NIL'
    assert doc[1].ent_kb_id_ == ''
    assert doc[2].ent_kb_id_ == 'Q2'

    def get_lowercased_candidates(kb, span):
        return kb.get_alias_candidates(span.text.lower())

    def get_lowercased_candidates_batch(kb, spans):
        return [get_lowercased_candidates(kb, span) for span in spans]

    @registry.misc('spacy.LowercaseCandidateGenerator.v1')
    def create_candidates() -> Callable[[InMemoryLookupKB, 'Span'], Iterable[Candidate]]:
        return get_lowercased_candidates

    @registry.misc('spacy.LowercaseCandidateBatchGenerator.v1')
    def create_candidates_batch() -> Callable[[InMemoryLookupKB, Iterable['Span']], Iterable[Iterable[Candidate]]]:
        return get_lowercased_candidates_batch
    entity_linker = nlp.replace_pipe('entity_linker', 'entity_linker', config={'incl_context': False, 'get_candidates': {'@misc': 'spacy.LowercaseCandidateGenerator.v1'}, 'get_candidates_batch': {'@misc': 'spacy.LowercaseCandidateBatchGenerator.v1'}})
    entity_linker.set_kb(create_kb)
    doc = nlp(text)
    assert doc[0].ent_kb_id_ == 'Q2'
    assert doc[1].ent_kb_id_ == ''
    assert doc[2].ent_kb_id_ == 'Q2'