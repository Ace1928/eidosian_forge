import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def filtered_forecasts_error_cov(self):
    if self._filtered_forecasts_cov is None:
        self._filtered_forecasts, self._filtered_forecasts_cov = self._compute_forecasts(self.filtered_state, self.filtered_state_cov)
    return self._filtered_forecasts_cov