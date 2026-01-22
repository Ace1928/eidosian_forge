import os
from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format
import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
from cirq_google.serialization.arg_func_langs import arg_to_proto
def test_any_none():
    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    assert not PhasedFSimCharacterization(**characterization_angles).any_none()
    for angle in characterization_angles:
        none_angles = dict(characterization_angles)
        del none_angles[angle]
        assert PhasedFSimCharacterization(**none_angles).any_none()