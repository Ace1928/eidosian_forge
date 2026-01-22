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
def draws(self, *, inc_warmup: bool=False, inc_iterations: bool=False, concat_chains: bool=False, inc_sample: bool=False) -> np.ndarray:
    """
        Returns a numpy.ndarray over the generated quantities draws from
        all chains which is stored column major so that the values
        for a parameter are contiguous in memory, likewise all draws from
        a chain are contiguous.  By default, returns a 3D array arranged
        (draws, chains, columns); parameter ``concat_chains=True`` will
        return a 2D array where all chains are flattened into a single column,
        preserving chain order, so that given M chains of N draws,
        the first N draws are from chain 1, ..., and the the last N draws
        are from chain M.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the output, i.e., the sampler was run with ``save_warmup=True``,
            then the warmup draws are included.  Default value is ``False``.

        :param concat_chains: When ``True`` return a 2D array flattening all
            all draws from all chains.  Default value is ``False``.

        :param inc_sample: When ``True`` include all columns in the previous_fit
            draws array as well, excepting columns for variables already present
            in the generated quantities drawset. Default value is ``False``.

        See Also
        --------
        CmdStanGQ.draws_pd
        CmdStanGQ.draws_xr
        CmdStanMCMC.draws
        """
    self._assemble_generated_quantities()
    inc_warmup |= inc_iterations
    if inc_warmup:
        if isinstance(self.previous_fit, CmdStanMCMC) and (not self.previous_fit._save_warmup):
            get_logger().warning('Sample doesn\'t contain draws from warmup iterations, rerun sampler with "save_warmup=True".')
        elif isinstance(self.previous_fit, CmdStanMLE) and (not self.previous_fit._save_iterations):
            get_logger().warning('MLE doesn\'t contain draws from pre-convergence iterations, rerun optimization with "save_iterations=True".')
        elif isinstance(self.previous_fit, CmdStanVB):
            get_logger().warning('Variational fit doesn\'t make sense with argument "inc_warmup=True"')
    if inc_sample:
        cols_1 = self.previous_fit.column_names
        cols_2 = self.column_names
        dups = [item for item, count in Counter(cols_1 + cols_2).items() if count > 1]
        drop_cols: List[int] = []
        for dup in dups:
            drop_cols.extend(self.previous_fit._metadata.stan_vars[dup].columns())
    start_idx, _ = self._draws_start(inc_warmup)
    previous_draws = self._previous_draws(True)
    if concat_chains and inc_sample:
        return flatten_chains(np.dstack((np.delete(previous_draws, drop_cols, axis=1), self._draws))[start_idx:, :, :])
    if concat_chains:
        return flatten_chains(self._draws[start_idx:, :, :])
    if inc_sample:
        return np.dstack((np.delete(previous_draws, drop_cols, axis=1), self._draws))[start_idx:, :, :]
    return self._draws[start_idx:, :, :]