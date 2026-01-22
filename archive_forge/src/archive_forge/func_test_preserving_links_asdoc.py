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
def test_preserving_links_asdoc(nlp):
    """Test that Span.as_doc preserves the existing entity links"""
    vector_length = 1

    def create_kb(vocab):
        mykb = InMemoryLookupKB(vocab, entity_vector_length=vector_length)
        mykb.add_entity(entity='Q1', freq=19, entity_vector=[1])
        mykb.add_entity(entity='Q2', freq=8, entity_vector=[1])
        mykb.add_alias(alias='Boston', entities=['Q1'], probabilities=[0.7])
        mykb.add_alias(alias='Denver', entities=['Q2'], probabilities=[0.6])
        return mykb
    nlp.add_pipe('sentencizer')
    patterns = [{'label': 'GPE', 'pattern': 'Boston'}, {'label': 'GPE', 'pattern': 'Denver'}]
    ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    config = {'incl_prior': False}
    entity_linker = nlp.add_pipe('entity_linker', config=config, last=True)
    entity_linker.set_kb(create_kb)
    nlp.initialize()
    assert entity_linker.model.get_dim('nO') == vector_length
    text = 'She lives in Boston. He lives in Denver.'
    doc = nlp(text)
    for ent in doc.ents:
        orig_text = ent.text
        orig_kb_id = ent.kb_id_
        sent_doc = ent.sent.as_doc()
        for s_ent in sent_doc.ents:
            if s_ent.text == orig_text:
                assert s_ent.kb_id_ == orig_kb_id