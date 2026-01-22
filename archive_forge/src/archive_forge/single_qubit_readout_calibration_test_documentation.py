from typing import Sequence
import pytest
import numpy as np
import cirq
Sampler that flips some bits upon readout.

        Args:
            p0: Probability of flipping a 0 to a 1.
            p1: Probability of flipping a 1 to a 0.
            seed: A seed for the pseudorandom number generator.
        