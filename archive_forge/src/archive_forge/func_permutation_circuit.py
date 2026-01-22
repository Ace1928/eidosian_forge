from __future__ import annotations
import logging
from collections.abc import Mapping
import numpy as np
import rustworkx as rx
from .types import Swap, Permutation
from .util import PermutationCircuit, permutation_circuit
def permutation_circuit(self, permutation: Permutation, trials: int=4) -> PermutationCircuit:
    """Perform an approximately optimal Token Swapping algorithm to implement the permutation.

        Args:
          permutation: The partial mapping to implement in swaps.
          trials: The number of trials to try to perform the mapping. Minimize over the trials.

        Returns:
          The circuit to implement the permutation
        """
    sequential_swaps = self.map(permutation, trials=trials)
    parallel_swaps = [[swap] for swap in sequential_swaps]
    return permutation_circuit(parallel_swaps)