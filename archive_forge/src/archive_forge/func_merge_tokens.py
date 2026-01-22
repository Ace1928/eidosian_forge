from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torchaudio._extension import fail_if_no_align
def merge_tokens(tokens: Tensor, scores: Tensor, blank: int=0) -> List[TokenSpan]:
    """Removes repeated tokens and blank tokens from the given CTC token sequence.

    Args:
        tokens (Tensor): Alignment tokens (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`.
        scores (Tensor): Alignment scores (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`. When computing the token-size score, the given score is averaged
            across the corresponding time span.

    Returns:
        list of TokenSpan

    Example:
        >>> aligned_tokens, scores = forced_align(emission, targets, input_lengths, target_lengths)
        >>> token_spans = merge_tokens(aligned_tokens[0], scores[0])
    """
    if tokens.ndim != 1 or scores.ndim != 1:
        raise ValueError('`tokens` and `scores` must be 1D Tensor.')
    if len(tokens) != len(scores):
        raise ValueError('`tokens` and `scores` must be the same length.')
    diff = torch.diff(tokens, prepend=torch.tensor([-1], device=tokens.device), append=torch.tensor([-1], device=tokens.device))
    changes_wo_blank = torch.nonzero(diff != 0).squeeze().tolist()
    tokens = tokens.tolist()
    spans = [TokenSpan(token=token, start=start, end=end, score=scores[start:end].mean().item()) for start, end in zip(changes_wo_blank[:-1], changes_wo_blank[1:]) if (token := tokens[start]) != blank]
    return spans