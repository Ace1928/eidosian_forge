from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def hamiltonian_obs(self):
    """Hamiltonian observable matching ``use_csingle`` precision."""
    if self._use_mpi:
        return self.hamiltonian_mpi_c64 if self.use_csingle else self.hamiltonian_mpi_c128
    return self.hamiltonian_c64 if self.use_csingle else self.hamiltonian_c128