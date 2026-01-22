import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def filtered_signal(self):
    if self._filtered_signal is None:
        self._filtered_signal, self._filtered_signal_cov = self._compute_forecasts(self.filtered_state, self.filtered_state_cov, signal_only=True)
    return self._filtered_signal