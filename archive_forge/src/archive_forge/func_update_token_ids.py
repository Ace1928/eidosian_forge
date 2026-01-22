import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
@torch.inference_mode()
def update_token_ids(token_ids: torch.Tensor, next_token_ids: torch.Tensor, ancestors: torch.Tensor) -> torch.Tensor:
    """Append the sampled tokens to the running sequence of tokens.

    Parameters
    ----------
    token_ids
        The current token sequences
    next_token_ids
        The tokens that were just generated and that we need to append
        to the existing sequences.
    ancestors
        The sequences to which the token ids need to be added.

    Returns
    -------
    A new sequence of token ids that contains the tokens that were
    just generated.

    """
    token_ids = torch.index_select(token_ids, 0, ancestors)
    return torch.concatenate([token_ids, next_token_ids], dim=-1)