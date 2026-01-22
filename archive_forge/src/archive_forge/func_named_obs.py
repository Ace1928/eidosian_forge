from typing import List, Tuple
import numpy as np
from pennylane import (
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap
from pennylane import matrix, DeviceError
@property
def named_obs(self):
    """Named observable matching ``use_csingle`` precision."""
    if self._use_mpi:
        return self.named_obs_mpi_c64 if self.use_csingle else self.named_obs_mpi_c128
    return self.named_obs_c64 if self.use_csingle else self.named_obs_c128