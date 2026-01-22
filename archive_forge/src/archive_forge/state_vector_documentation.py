import warnings
from typing import cast, Sequence, Union, List, Tuple, Dict, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
Compute an expectation value for an operator and a circuit via tensor
    contraction.

    This will give up if it looks like the computation will take too much RAM.
    