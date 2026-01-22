import itertools
from typing import Dict, Iterator, List, Optional, Sequence, cast
import numpy as np

        Sample bitstrings from the distribution defined by the wavefunction.

        :param n_samples: The number of bitstrings to sample
        :return: An array of shape (n_samples, n_qubits)
        