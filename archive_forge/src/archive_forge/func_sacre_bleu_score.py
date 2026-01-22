import os
import re
import tempfile
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Sequence, Type
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import (
def sacre_bleu_score(preds: Sequence[str], target: Sequence[Sequence[str]], n_gram: int=4, smooth: bool=False, tokenize: _TokenizersLiteral='13a', lowercase: bool=False, weights: Optional[Sequence[float]]=None) -> Tensor:
    """Calculate `BLEU score`_ [1] of machine translated text with one or more references.

    This implementation follows the behaviour of SacreBLEU [2] implementation from https://github.com/mjpost/sacrebleu.

    Args:
        preds: An iterable of machine translated corpus
        target: An iterable of iterables of reference corpus
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing - see [2]
        tokenize: Tokenization technique to be used. Choose between ``'none'``, ``'13a'``, ``'zh'``, ``'intl'``,
            ``'char'``, ``'ja-mecab'``, ``'ko-mecab'``, ``'flores101'`` and ``'flores200'``.
        lowercase: If ``True``, BLEU score over lowercased text is calculated.
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Return:
        Tensor with BLEU Score

    Raises:
        ValueError: If ``preds`` and ``target`` corpus have different lengths.
        ValueError: If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.functional.text import sacre_bleu_score
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> sacre_bleu_score(preds, target)
        tensor(0.7598)

    References:
        [1] BLEU: a Method for Automatic Evaluation of Machine Translation by Papineni,
        Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu `BLEU`_

        [2] A Call for Clarity in Reporting BLEU Scores by Matt Post.

        [3] Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
        and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och `Machine Translation Evolution`_

    """
    if len(preds) != len(target):
        raise ValueError(f'Corpus has different size {len(preds)} != {len(target)}')
    if weights is not None and len(weights) != n_gram:
        raise ValueError(f'List of weights has different weights than `n_gram`: {len(weights)} != {n_gram}')
    if weights is None:
        weights = [1.0 / n_gram] * n_gram
    numerator = torch.zeros(n_gram)
    denominator = torch.zeros(n_gram)
    preds_len = tensor(0.0)
    target_len = tensor(0.0)
    tokenize_fn = partial(_SacreBLEUTokenizer.tokenize, tokenize=tokenize, lowercase=lowercase)
    preds_len, target_len = _bleu_score_update(preds, target, numerator, denominator, preds_len, target_len, n_gram, tokenize_fn)
    return _bleu_score_compute(preds_len, target_len, numerator, denominator, n_gram, weights, smooth)