from typing import Any, Dict, Sequence, Union
import cmath
import math
import cirq
from cirq import protocols
from cirq._doc import document
import numpy as np
@property
def phases(self) -> Sequence[float]:
    return [self.phi0, self.phi1]