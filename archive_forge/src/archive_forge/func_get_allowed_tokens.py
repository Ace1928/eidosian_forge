import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def get_allowed_tokens(fsms: List['Guide'], fsm_states: List[int]) -> torch.Tensor:
    """Get the new instructions for each sequence from the finite-state machine.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    A nested list that contains the ids of the logits to keep.

    """
    return [fsm.get_next_instruction(state).tokens for fsm, state in zip(fsms, fsm_states)]