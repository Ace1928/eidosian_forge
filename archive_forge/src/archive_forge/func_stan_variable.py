from collections import Counter
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import build_xarray_data, flatten_chains, get_logger
from cmdstanpy.utils.stancsv import scan_generic_csv
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB
def stan_variable(self, var: str, **kwargs: bool) -> np.ndarray:
    """
        Return a numpy.ndarray which contains the set of draws
        for the named Stan program variable.  Flattens the chains,
        leaving the draws in chain order.  The first array dimension,
        corresponds to number of draws in the sample.
        The remaining dimensions correspond to
        the shape of the Stan program variable.

        Underlyingly draws are in chain order, i.e., for a sample with
        N chains of M draws each, the first M array elements are from chain 1,
        the next M are from chain 2, and the last M elements are from chain N.

        * If the variable is a scalar variable, the return array has shape
          ( draws * chains, 1).
        * If the variable is a vector, the return array has shape
          ( draws * chains, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws * chains, size(dim 1), size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws * chains, size(dim 1), ..., size(dim N))

        For example, if the Stan program variable ``theta`` is a 3x3 matrix,
        and the sample consists of 4 chains with 1000 post-warmup draws,
        this function will return a numpy.ndarray with shape (4000,3,3).

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param kwargs: Additional keyword arguments are passed to the underlying
            fit's ``stan_variable`` method if the variable is not a generated
            quantity.

        See Also
        --------
        CmdStanGQ.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanLaplace.stan_variable
        """
    model_var_names = self.previous_fit._metadata.stan_vars.keys()
    gq_var_names = self._metadata.stan_vars.keys()
    if not (var in model_var_names or var in gq_var_names):
        raise ValueError(f'Unknown variable name: {var}\nAvailable variables are ' + ', '.join(model_var_names | gq_var_names))
    if var not in gq_var_names:
        return np.atleast_1d(self.previous_fit.stan_variable(var, **kwargs))
    self._assemble_generated_quantities()
    draw1, _ = self._draws_start(inc_warmup=kwargs.get('inc_warmup', False) or kwargs.get('inc_iterations', False))
    draws = flatten_chains(self._draws[draw1:])
    out: np.ndarray = self._metadata.stan_vars[var].extract_reshape(draws)
    return out