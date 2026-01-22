from __future__ import (absolute_import, division, print_function)
import numpy as np
from .plotting import plot_result, plot_phase_plane, info_vlines
from .util import import_
def plot_invariant_violations(self, **kwargs):
    viol = self.calc_invariant_violations()
    abs_viol = np.abs(viol)
    invar_names = self.odesys.all_invariant_names()
    return self._plot(plot_result, x=self._internal('xout'), y=abs_viol, names=invar_names, latex_names=kwargs.pop('latex_names', invar_names), indices=None, **kwargs)