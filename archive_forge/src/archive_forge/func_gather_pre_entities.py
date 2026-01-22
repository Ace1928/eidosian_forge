import types
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
from ..models.bert.tokenization_bert import BasicTokenizer
from ..utils import (
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args
def gather_pre_entities(self, sentence: str, input_ids: np.ndarray, scores: np.ndarray, offset_mapping: Optional[List[Tuple[int, int]]], special_tokens_mask: np.ndarray, aggregation_strategy: AggregationStrategy) -> List[dict]:
    """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
    pre_entities = []
    for idx, token_scores in enumerate(scores):
        if special_tokens_mask[idx]:
            continue
        word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
        if offset_mapping is not None:
            start_ind, end_ind = offset_mapping[idx]
            if not isinstance(start_ind, int):
                if self.framework == 'pt':
                    start_ind = start_ind.item()
                    end_ind = end_ind.item()
            word_ref = sentence[start_ind:end_ind]
            if getattr(self.tokenizer, '_tokenizer', None) and getattr(self.tokenizer._tokenizer.model, 'continuing_subword_prefix', None):
                is_subword = len(word) != len(word_ref)
            else:
                if aggregation_strategy in {AggregationStrategy.FIRST, AggregationStrategy.AVERAGE, AggregationStrategy.MAX}:
                    warnings.warn('Tokenizer does not support real words, using fallback heuristic', UserWarning)
                is_subword = start_ind > 0 and ' ' not in sentence[start_ind - 1:start_ind + 1]
            if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                word = word_ref
                is_subword = False
        else:
            start_ind = None
            end_ind = None
            is_subword = False
        pre_entity = {'word': word, 'scores': token_scores, 'start': start_ind, 'end': end_ind, 'index': idx, 'is_subword': is_subword}
        pre_entities.append(pre_entity)
    return pre_entities