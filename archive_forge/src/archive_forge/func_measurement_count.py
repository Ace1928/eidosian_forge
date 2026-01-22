import math
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pytest
import sympy
import cirq
@property
def measurement_count(self):
    return self._state.measurement_count