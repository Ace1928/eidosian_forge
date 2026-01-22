from abc import abstractmethod
from typing import (
from .config import registry
from .types import Floats2d, Ints1d
from .util import get_array_module, to_categorical
def get_similarity(self, guesses: Floats2d, truths: Floats2d) -> float:
    if guesses.shape != truths.shape:
        err = f'Cannot calculate cosine similarity: mismatched shapes: {guesses.shape} vs {truths.shape}.'
        raise ValueError(err)
    xp = get_array_module(guesses)
    yh = guesses + 1e-08
    y = truths + 1e-08
    norm_yh = xp.linalg.norm(yh, axis=1, keepdims=True)
    norm_y = xp.linalg.norm(y, axis=1, keepdims=True)
    mul_norms = norm_yh * norm_y
    cosine = (yh * y).sum(axis=1, keepdims=True) / mul_norms
    return cosine