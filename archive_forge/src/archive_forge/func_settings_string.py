import json
import logging
import warnings
from json import JSONEncoder
from typing import (
from pyquil.experiment._calibration import CalibrationMethod
from pyquil.experiment._memory import (
from pyquil.experiment._program import (
from pyquil.experiment._result import ExperimentResult
from pyquil.experiment._setting import ExperimentSetting, _OneQState, TensorProductState
from pyquil.experiment._symmetrization import SymmetrizationLevel
from pyquil.gates import RESET
from pyquil.paulis import PauliTerm, is_identity
from pyquil.quil import Program
from pyquil.quilbase import Reset, ResetQubit
def settings_string(self, abbrev_after: Optional[int]=None) -> str:
    setting_strs = list(self.setting_strings())
    if abbrev_after is not None and len(setting_strs) > abbrev_after:
        first_n = abbrev_after // 2
        last_n = abbrev_after - first_n
        excluded = len(setting_strs) - abbrev_after
        setting_strs = setting_strs[:first_n] + [f'... {excluded} settings not shown ...'] + setting_strs[-last_n:]
    return '   ' + '\n   '.join(setting_strs)