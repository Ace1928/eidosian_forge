from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane.measurements import VarianceMP
from pennylane import transform
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from .finite_difference import finite_diff
from .general_shift_rules import (
from .gradient_transform import (
Auxiliary function for post-processing one batch of supported and unsupported gradients corresponding to
            finite shot execution.

            If the device used a shot vector, gradients corresponding to a single component of the shot vector should be
            passed to this aux function.
            