from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def chrf_score(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], n_char_order: int=6, n_word_order: int=2, beta: float=2.0, lowercase: bool=False, whitespace: bool=False, return_sentence_level_score: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Calculate `chrF score`_  of machine translated text with one or more references.

    This implementation supports both chrF score computation introduced in [1] and chrF++ score introduced in
    `chrF++ score`_. This implementation follows the implementations from https://github.com/m-popovic/chrF and
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/chrf.py.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        n_char_order:
            A character n-gram order. If `n_char_order=6`, the metrics refers to the official chrF/chrF++.
        n_word_order:
            A word n-gram order. If `n_word_order=2`, the metric refers to the official chrF++. If `n_word_order=0`, the
            metric is equivalent to the original chrF.
        beta:
            A parameter determining an importance of recall w.r.t. precision. If `beta=1`, their importance is equal.
        lowercase: An indication whether to enable case-insensitivity.
        whitespace: An indication whether to keep whitespaces during character n-gram extraction.
        return_sentence_level_score: An indication whether a sentence-level chrF/chrF++ score to be returned.

    Return:
        A corpus-level chrF/chrF++ score.
        (Optionally) A list of sentence-level chrF/chrF++ scores if `return_sentence_level_score=True`.

    Raises:
        ValueError:
            If ``n_char_order`` is not an integer greater than or equal to 1.
        ValueError:
            If ``n_word_order`` is not an integer greater than or equal to 0.
        ValueError:
            If ``beta`` is smaller than 0.

    Example:
        >>> from torchmetrics.functional.text import chrf_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> chrf_score(preds, target)
        tensor(0.8640)

    References:
        [1] chrF: character n-gram F-score for automatic MT evaluation by Maja Popović `chrF score`_

        [2] chrF++: words helping character n-grams by Maja Popović `chrF++ score`_

    """
    if not isinstance(n_char_order, int) or n_char_order < 1:
        raise ValueError('Expected argument `n_char_order` to be an integer greater than or equal to 1.')
    if not isinstance(n_word_order, int) or n_word_order < 0:
        raise ValueError('Expected argument `n_word_order` to be an integer greater than or equal to 0.')
    if beta < 0:
        raise ValueError('Expected argument `beta` to be greater than 0.')
    n_order = float(n_char_order + n_word_order)
    total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams = _prepare_n_grams_dicts(n_char_order, n_word_order)
    sentence_chrf_score: Optional[List[Tensor]] = [] if return_sentence_level_score else None
    total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams, sentence_chrf_score = _chrf_score_update(preds, target, total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams, n_char_order, n_word_order, n_order, beta, lowercase, whitespace, sentence_chrf_score)
    chrf_f_score = _chrf_score_compute(total_preds_char_n_grams, total_preds_word_n_grams, total_target_char_n_grams, total_target_word_n_grams, total_matching_char_n_grams, total_matching_word_n_grams, n_order, beta)
    if sentence_chrf_score:
        return (chrf_f_score, torch.cat(sentence_chrf_score))
    return chrf_f_score