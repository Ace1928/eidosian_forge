import warnings
from .base_scheduler import BaseScheduler
@staticmethod
def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero=False):
    """"Computes the current level of sparsity.

        Based on https://arxiv.org/pdf/1710.01878.pdf

        Args:
            s_0: Initial level of sparsity, :math:`s_i`
            s_f: Target level of sparsity, :math:`s_f`
            t: Current step, :math:`t`
            t_0: Initial step, :math:`t_0`
            dt: Pruning frequency, :math:`\\Delta T`
            n: Pruning steps, :math:`n`
            initially_zero: Sets the level of sparsity to 0 before t_0.
                If False, sets to s_0

        Returns:
            The sparsity level :math:`s_t` at the current step :math:`t`
        """
    if initially_zero and t < t_0:
        return 0
    s_t = s_f + (s_0 - s_f) * (1.0 - (t - t_0) / (dt * n)) ** 3
    s_t = _clamp(s_t, s_0, s_f)
    return s_t