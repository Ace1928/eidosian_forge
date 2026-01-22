import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def zeta_chi_gamma_correction_override(self) -> PhasedFSimCharacterization:
    """Gives a PhasedFSimCharacterization that can be used to override characterization after
        correcting for zeta, chi and gamma angles.
        """
    return PhasedFSimCharacterization(zeta=0.0 if self.characterize_zeta else None, chi=0.0 if self.characterize_chi else None, gamma=0.0 if self.characterize_gamma else None)