import random
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import srsly
from thinc.api import Config, CosineDistance, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from .. import util
from ..errors import Errors
from ..kb import Candidate, KnowledgeBase
from ..language import Language
from ..ml import empty_kb
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example, validate_examples, validate_get_examples
from ..util import SimpleFrozenList, registry
from ..vocab import Vocab
from .legacy.entity_linker import EntityLinker_v1
from .pipe import deserialize_config
from .trainable_pipe import TrainablePipe
@Language.factory('entity_linker', requires=['doc.ents', 'doc.sents', 'token.ent_iob', 'token.ent_type'], assigns=['token.ent_kb_id'], default_config={'model': DEFAULT_NEL_MODEL, 'labels_discard': [], 'n_sents': 0, 'incl_prior': True, 'incl_context': True, 'entity_vector_length': 64, 'get_candidates': {'@misc': 'spacy.CandidateGenerator.v1'}, 'get_candidates_batch': {'@misc': 'spacy.CandidateBatchGenerator.v1'}, 'generate_empty_kb': {'@misc': 'spacy.EmptyKB.v2'}, 'overwrite': True, 'scorer': {'@scorers': 'spacy.entity_linker_scorer.v1'}, 'use_gold_ents': True, 'candidates_batch_size': 1, 'threshold': None}, default_score_weights={'nel_micro_f': 1.0, 'nel_micro_r': None, 'nel_micro_p': None})
def make_entity_linker(nlp: Language, name: str, model: Model, *, labels_discard: Iterable[str], n_sents: int, incl_prior: bool, incl_context: bool, entity_vector_length: int, get_candidates: Callable[[KnowledgeBase, Span], Iterable[Candidate]], get_candidates_batch: Callable[[KnowledgeBase, Iterable[Span]], Iterable[Iterable[Candidate]]], generate_empty_kb: Callable[[Vocab, int], KnowledgeBase], overwrite: bool, scorer: Optional[Callable], use_gold_ents: bool, candidates_batch_size: int, threshold: Optional[float]=None):
    """Construct an EntityLinker component.

    model (Model[List[Doc], Floats2d]): A model that learns document vector
        representations. Given a batch of Doc objects, it should return a single
        array, with one row per item in the batch.
    labels_discard (Iterable[str]): NER labels that will automatically get a "NIL" prediction.
    n_sents (int): The number of neighbouring sentences to take into account.
    incl_prior (bool): Whether or not to include prior probabilities from the KB in the model.
    incl_context (bool): Whether or not to include the local context in the model.
    entity_vector_length (int): Size of encoding vectors in the KB.
    get_candidates (Callable[[KnowledgeBase, Span], Iterable[Candidate]]): Function that
        produces a list of candidates, given a certain knowledge base and a textual mention.
    get_candidates_batch (
        Callable[[KnowledgeBase, Iterable[Span]], Iterable[Iterable[Candidate]]], Iterable[Candidate]]
        ): Function that produces a list of candidates, given a certain knowledge base and several textual mentions.
    generate_empty_kb (Callable[[Vocab, int], KnowledgeBase]): Callable returning empty KnowledgeBase.
    scorer (Optional[Callable]): The scoring method.
    use_gold_ents (bool): Whether to copy entities from gold docs or not. If false, another
        component must provide entity annotations.
    candidates_batch_size (int): Size of batches for entity candidate generation.
    threshold (Optional[float]): Confidence threshold for entity predictions. If confidence is below the threshold,
        prediction is discarded. If None, predictions are not filtered by any threshold.
    """
    if not model.attrs.get('include_span_maker', False):
        return EntityLinker_v1(nlp.vocab, model, name, labels_discard=labels_discard, n_sents=n_sents, incl_prior=incl_prior, incl_context=incl_context, entity_vector_length=entity_vector_length, get_candidates=get_candidates, overwrite=overwrite, scorer=scorer)
    return EntityLinker(nlp.vocab, model, name, labels_discard=labels_discard, n_sents=n_sents, incl_prior=incl_prior, incl_context=incl_context, entity_vector_length=entity_vector_length, get_candidates=get_candidates, get_candidates_batch=get_candidates_batch, generate_empty_kb=generate_empty_kb, overwrite=overwrite, scorer=scorer, use_gold_ents=use_gold_ents, candidates_batch_size=candidates_batch_size, threshold=threshold)