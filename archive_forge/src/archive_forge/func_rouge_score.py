import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def rouge_score(preds: Union[str, Sequence[str]], target: Union[str, Sequence[str], Sequence[Sequence[str]]], accumulate: Literal['avg', 'best']='best', use_stemmer: bool=False, normalizer: Optional[Callable[[str], str]]=None, tokenizer: Optional[Callable[[str], Sequence[str]]]=None, rouge_keys: Union[str, Tuple[str, ...]]=('rouge1', 'rouge2', 'rougeL', 'rougeLsum')) -> Dict[str, Tensor]:
    """Calculate `Calculate Rouge Score`_ , used for automatic summarization.

    Args:
        preds: An iterable of predicted sentences or a single predicted sentence.
        target:
            An iterable of iterables of target sentences or an iterable of target sentences or a single target sentence.
        accumulate:
            Useful in case of multi-reference rouge score.

            - ``avg`` takes the avg of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.

        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer: A user's own tokenizer function. If this is ``None``, splitting by spaces is default
            This function must take a ``str`` and return ``Sequence[str]``
        rouge_keys: A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.

    Return:
        Python dictionary of rouge scores for each input rouge key.

    Example:
        >>> from torchmetrics.functional.text.rouge import rouge_score
        >>> preds = "My name is John"
        >>> target = "Is your name John"
        >>> from pprint import pprint
        >>> pprint(rouge_score(preds, target))
        {'rouge1_fmeasure': tensor(0.7500),
         'rouge1_precision': tensor(0.7500),
         'rouge1_recall': tensor(0.7500),
         'rouge2_fmeasure': tensor(0.),
         'rouge2_precision': tensor(0.),
         'rouge2_recall': tensor(0.),
         'rougeL_fmeasure': tensor(0.5000),
         'rougeL_precision': tensor(0.5000),
         'rougeL_recall': tensor(0.5000),
         'rougeLsum_fmeasure': tensor(0.5000),
         'rougeLsum_precision': tensor(0.5000),
         'rougeLsum_recall': tensor(0.5000)}


    Raises:
        ModuleNotFoundError:
            If the python package ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin. https://aclanthology.org/W04-1013/

    """
    if use_stemmer:
        if not _NLTK_AVAILABLE:
            raise ModuleNotFoundError('Stemmer requires that `nltk` is installed. Use `pip install nltk`.')
        import nltk
    stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None
    if not isinstance(rouge_keys, tuple):
        rouge_keys = (rouge_keys,)
    for key in rouge_keys:
        if key not in ALLOWED_ROUGE_KEYS:
            raise ValueError(f'Got unknown rouge key {key}. Expected to be one of {list(ALLOWED_ROUGE_KEYS.keys())}')
    rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]
    if isinstance(target, list) and all((isinstance(tgt, str) for tgt in target)):
        target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [[target]]
    sentence_results: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(preds, target, rouge_keys_values, stemmer=stemmer, normalizer=normalizer, tokenizer=tokenizer, accumulate=accumulate)
    output: Dict[str, List[Tensor]] = {f'rouge{rouge_key}_{tp}': [] for rouge_key in rouge_keys_values for tp in ['fmeasure', 'precision', 'recall']}
    for rouge_key, metrics in sentence_results.items():
        for metric in metrics:
            for tp, value in metric.items():
                output[f'rouge{rouge_key}_{tp}'].append(value)
    return _rouge_score_compute(output)