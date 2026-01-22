import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def benchmark_2q_xeb_fidelities(sampled_df: pd.DataFrame, circuits: Sequence['cirq.Circuit'], cycle_depths: Optional[Sequence[int]]=None, param_resolver: 'cirq.ParamResolverOrSimilarType'=None, pool: Optional['multiprocessing.pool.Pool']=None) -> pd.DataFrame:
    """Simulate and benchmark two-qubit XEB circuits.

    This uses the estimator from
    `cirq.experiments.fidelity_estimation.least_squares_xeb_fidelity_from_expectations`, but
    adapted for use on pandas DataFrames for efficient vectorized operation.

    Args:
        sampled_df: The sampled results to benchmark. This is likely produced by a call to
            `sample_2q_xeb_circuits`.
        circuits: The library of circuits corresponding to the sampled results in `sampled_df`.
        cycle_depths: The sequence of cycle depths to benchmark the circuits. If not provided,
            we use the cycle depths found in `sampled_df`. All requested `cycle_depths` must be
            present in `sampled_df`.
        param_resolver: If circuits contain parameters, resolve according to this ParamResolver
            prior to simulation
        pool: If provided, execute the simulations in parallel.

    Returns:
        A DataFrame with columns 'cycle_depth' and 'fidelity'.

    Raises:
        ValueError: If `cycle_depths` is not a non-empty array or if the `cycle_depths` provided
            includes some values not available in `sampled_df`.
    """
    sampled_cycle_depths = sampled_df.index.get_level_values('cycle_depth').drop_duplicates().sort_values()
    if cycle_depths is not None:
        if len(cycle_depths) == 0:
            raise ValueError('`cycle_depths` should be a non-empty array_like')
        not_in_sampled = np.setdiff1d(cycle_depths, sampled_cycle_depths)
        if len(not_in_sampled) > 0:
            raise ValueError(f'The `cycle_depths` provided include some not available in `sampled_df`: {not_in_sampled}')
        sim_cycle_depths = cycle_depths
    else:
        sim_cycle_depths = sampled_cycle_depths
    simulated_df = simulate_2q_xeb_circuits(circuits=circuits, cycle_depths=sim_cycle_depths, param_resolver=param_resolver, pool=pool)
    df = sampled_df.join(simulated_df, how='inner').reset_index()
    D = 4
    pure_probs = np.array(df['pure_probs'].to_list())
    sampled_probs = np.array(df['sampled_probs'].to_list())
    df['e_u'] = np.sum(pure_probs ** 2, axis=1)
    df['u_u'] = np.sum(pure_probs, axis=1) / D
    df['m_u'] = np.sum(pure_probs * sampled_probs, axis=1)
    df['y'] = df['m_u'] - df['u_u']
    df['x'] = df['e_u'] - df['u_u']
    df['numerator'] = df['x'] * df['y']
    df['denominator'] = df['x'] ** 2

    def per_cycle_depth(df):
        """This function is applied per cycle_depth in the following groupby aggregation."""
        fid_lsq = df['numerator'].sum() / df['denominator'].sum()
        ret = {'fidelity': fid_lsq}

        def _try_keep(k):
            """If all the values for a key `k` are the same in this group, we can keep it."""
            if k not in df.columns:
                return
            vals = df[k].unique()
            if len(vals) == 1:
                ret[k] = vals[0]
            else:
                raise AssertionError(f'When computing per-cycle-depth fidelity, multiple values for {k} were grouped together: {vals}')
        _try_keep('pair')
        return pd.Series(ret)
    if 'pair_i' in df.columns:
        groupby_names = ['layer_i', 'pair_i', 'cycle_depth']
    else:
        groupby_names = ['cycle_depth']
    return df.groupby(groupby_names).apply(per_cycle_depth).reset_index()