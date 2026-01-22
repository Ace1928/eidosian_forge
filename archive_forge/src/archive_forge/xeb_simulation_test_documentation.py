import multiprocessing
from typing import Dict, Any, Optional
from typing import Sequence
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
Reference implementation for `simulate_2q_xeb_circuits` that
    does each circuit independently instead of using intermediate states.

    You can also try editing the helper function to use QSimSimulator() for
    benchmarking. This simulator does not support intermediate states, so
    you can't use it with the new functionality.
    https://github.com/quantumlib/qsim/issues/101
    