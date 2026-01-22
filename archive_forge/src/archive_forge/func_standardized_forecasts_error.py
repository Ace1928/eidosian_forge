import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@property
def standardized_forecasts_error(self):
    """
        Standardized forecast errors

        Notes
        -----
        The forecast errors produced by the Kalman filter are

        .. math::

            v_t \\sim N(0, F_t)

        Hypothesis tests are usually applied to the standardized residuals

        .. math::

            v_t^s = B_t v_t \\sim N(0, I)

        where :math:`B_t = L_t^{-1}` and :math:`F_t = L_t L_t'`; then
        :math:`F_t^{-1} = (L_t')^{-1} L_t^{-1} = B_t' B_t`; :math:`B_t`
        and :math:`L_t` are lower triangular. Finally,
        :math:`B_t v_t \\sim N(0, B_t F_t B_t')` and
        :math:`B_t F_t B_t' = L_t^{-1} L_t L_t' (L_t')^{-1} = I`.

        Thus we can rewrite :math:`v_t^s = L_t^{-1} v_t` or
        :math:`L_t v_t^s = v_t`; the latter equation is the form required to
        use a linear solver to recover :math:`v_t^s`. Since :math:`L_t` is
        lower triangular, we can use a triangular solver (?TRTRS).
        """
    if self._standardized_forecasts_error is None and (not self.memory_no_forecast):
        if self.k_endog == 1:
            self._standardized_forecasts_error = self.forecasts_error / self.forecasts_error_cov[0, 0, :] ** 0.5
        else:
            from scipy import linalg
            self._standardized_forecasts_error = np.zeros(self.forecasts_error.shape, dtype=self.dtype)
            for t in range(self.forecasts_error_cov.shape[2]):
                if self.nmissing[t] > 0:
                    self._standardized_forecasts_error[:, t] = np.nan
                if self.nmissing[t] < self.k_endog:
                    mask = ~self.missing[:, t].astype(bool)
                    F = self.forecasts_error_cov[np.ix_(mask, mask, [t])]
                    try:
                        upper, _ = linalg.cho_factor(F[:, :, 0])
                        self._standardized_forecasts_error[mask, t] = linalg.solve_triangular(upper, self.forecasts_error[mask, t], trans=1)
                    except linalg.LinAlgError:
                        self._standardized_forecasts_error[mask, t] = np.nan
    return self._standardized_forecasts_error