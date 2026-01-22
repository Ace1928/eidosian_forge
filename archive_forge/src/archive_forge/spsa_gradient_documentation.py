from typing import Sequence, Callable
from functools import partial
import numpy as np
import pennylane as qml
from pennylane import transform
from pennylane.gradients.gradient_transform import _contract_qjac_with_cjac
from pennylane.transforms.tape_expand import expand_invalid_trainable
from .finite_difference import _processing_fn, finite_diff_coeffs
from .gradient_transform import (
from .general_shift_rules import generate_multishifted_tapes
Auxiliary function for post-processing one batch of results corresponding to finite
        shots or a single component of a shot vector