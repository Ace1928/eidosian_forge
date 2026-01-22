from typing import Dict, List, Sequence
import numpy as np
import cirq
from cirq import protocols, value
from cirq.qis.clifford_tableau import CliffordTableau
from cirq.sim.clifford.clifford_tableau_simulation_state import CliffordTableauSimulationState
from cirq.work import sampler
Inits StabilizerSampler.

        Args:
            seed: The random seed or generator to use when sampling.
        