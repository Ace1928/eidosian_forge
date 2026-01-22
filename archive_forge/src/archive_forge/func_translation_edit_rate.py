import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def translation_edit_rate(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], normalize: bool=False, no_punctuation: bool=False, lowercase: bool=True, asian_support: bool=False, return_sentence_level_score: bool=False) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
    """Calculate Translation edit rate (`TER`_)  of machine translated text with one or more references.

    This implementation follows the implementations from
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/ter.py. The `sacrebleu` implementation is a
    near-exact reimplementation of the Tercom algorithm, produces identical results on all "sane" outputs.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        normalize: An indication whether a general tokenization to be applied.
        no_punctuation: An indication whteher a punctuation to be removed from the sentences.
        lowercase: An indication whether to enable case-insensitivity.
        asian_support: An indication whether asian characters to be processed.
        return_sentence_level_score: An indication whether a sentence-level TER to be returned.

    Return:
        A corpus-level translation edit rate (TER).
        (Optionally) A list of sentence-level translation_edit_rate (TER) if `return_sentence_level_score=True`.

    Example:
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> translation_edit_rate(preds, target)
        tensor(0.1538)

    References:
        [1] A Study of Translation Edit Rate with Targeted Human Annotation
        by Mathew Snover, Bonnie Dorr, Richard Schwartz, Linnea Micciulla and John Makhoul `TER`_

    """
    if not isinstance(normalize, bool):
        raise ValueError(f'Expected argument `normalize` to be of type boolean but got {normalize}.')
    if not isinstance(no_punctuation, bool):
        raise ValueError(f'Expected argument `no_punctuation` to be of type boolean but got {no_punctuation}.')
    if not isinstance(lowercase, bool):
        raise ValueError(f'Expected argument `lowercase` to be of type boolean but got {lowercase}.')
    if not isinstance(asian_support, bool):
        raise ValueError(f'Expected argument `asian_support` to be of type boolean but got {asian_support}.')
    tokenizer: _TercomTokenizer = _TercomTokenizer(normalize, no_punctuation, lowercase, asian_support)
    total_num_edits = tensor(0.0)
    total_tgt_length = tensor(0.0)
    sentence_ter: Optional[List[Tensor]] = [] if return_sentence_level_score else None
    total_num_edits, total_tgt_length, sentence_ter = _ter_update(preds, target, tokenizer, total_num_edits, total_tgt_length, sentence_ter)
    ter_score = _ter_compute(total_num_edits, total_tgt_length)
    if sentence_ter:
        return (ter_score, sentence_ter)
    return ter_score