import logging
from math import pi
from numbers import Complex
from typing import Callable, Generator, List, Mapping, Tuple, Optional, cast
import numpy as np
from pyquil.api import QuantumComputer
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator
from pyquil.experiment._group import (
from pyquil.experiment._main import (
from pyquil.experiment._result import ExperimentResult, ratio_variance
from pyquil.experiment._setting import (
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET, RX, RY, RZ, X
from pyquil.paulis import is_identity
from pyquil.quil import Program
from pyquil.quilatom import QubitDesignator

    Program required for calibration in a tomography-like experiment.

    :param tomo_experiment: A suite of tomographic observables
    :param ExperimentSetting: The particular tomographic observable to measure
    :param symmetrize_readout: Method used to symmetrize the readout errors (see docstring for
        `measure_observables` for more details)
    :param cablir_shots: number of shots to take in the measurement process
    :return: Program performing the calibration
    