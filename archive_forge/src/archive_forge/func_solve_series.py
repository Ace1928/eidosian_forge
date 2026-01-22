from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def solve_series(self, x0, params, varied_data, varied_idx, internal_x0=None, solver=None, propagate=True, **kwargs):
    """ Solve system for a set of parameters in which one is varied

        Parameters
        ----------
        x0 : array_like
            Guess (subject to ``self.post_processors``)
        params : array_like
            Parameter values
        vaired_data : array_like
            Numerical values of the varied parameter.
        varied_idx : int or str
            Index of the varied parameter (indexing starts at 0).
            If ``self.par_by_name`` this should be the name (str) of the varied
            parameter.
        internal_x0 : array_like (default: None)
            Guess (*not* subject to ``self.post_processors``).
            Overrides ``x0`` when given.
        solver : str or callback
            See :meth:`solve`.
        propagate : bool (default: True)
            Use last successful solution as ``x0`` in consecutive solves.
        \\*\\*kwargs :
            Keyword arguments pass along to :meth:`solve`.

        Returns
        -------
        xout : array
            Of shape ``(varied_data.size, x0.size)``.
        info_dicts : list of dictionaries
             Dictionaries each containing keys such as containing 'success', 'nfev', 'njev' etc.

        """
    if self.x_by_name and isinstance(x0, dict):
        x0 = [x0[k] for k in self.names]
    if self.par_by_name:
        if isinstance(params, dict):
            params = [params[k] for k in self.param_names]
        if isinstance(varied_idx, str):
            varied_idx = self.param_names.index(varied_idx)
    new_params = np.atleast_1d(np.array(params, dtype=np.float64))
    xout = np.empty((len(varied_data), len(x0)))
    self.internal_xout = np.empty_like(xout)
    self.internal_params_out = np.empty((len(varied_data), len(new_params)))
    info_dicts = []
    new_x0 = np.array(x0, dtype=np.float64)
    conds = kwargs.get('initial_conditions', None)
    for idx, value in enumerate(varied_data):
        try:
            new_params[varied_idx] = value
        except TypeError:
            new_params = value
        if conds is not None:
            kwargs['initial_conditions'] = conds
        x, info_dict = self.solve(new_x0, new_params, internal_x0, solver, **kwargs)
        if propagate:
            if info_dict['success']:
                try:
                    new_x0 = info_dict['x_vecs'][0]
                    internal_x0 = info_dict['internal_x_vecs'][0]
                    conds = info_dict['intermediate_info'][0].get('conditions', None)
                except:
                    new_x0 = x
                    internal_x0 = None
                    conds = info_dict.get('conditions', None)
        xout[idx, :] = x
        self.internal_xout[idx, :] = self.internal_x
        self.internal_params_out[idx, :] = self.internal_params
        info_dicts.append(info_dict)
    return (xout, info_dicts)