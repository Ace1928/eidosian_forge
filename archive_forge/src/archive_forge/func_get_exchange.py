from __future__ import annotations
import copy
import logging
from ast import literal_eval
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.json import MSONable, jsanitize
from monty.serialization import dumpfn
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_exchange(self):
    """
        Take Heisenberg Hamiltonian and corresponding energy for each row and
        solve for the exchange parameters.

        Returns:
            ex_params (dict): Exchange parameter values (meV/atom).
        """
    ex_mat = self.ex_mat
    E = ex_mat[['E']]
    j_names = [j for j in ex_mat.columns if j != 'E']
    if len(j_names) < 3:
        j_avg = self.estimate_exchange()
        ex_params = {'<J>': j_avg}
        self.ex_params = ex_params
        return ex_params
    H = np.array(ex_mat.loc[:, ex_mat.columns != 'E'].values).astype(float)
    H_inv = np.linalg.inv(H)
    j_ij = np.dot(H_inv, E)
    j_ij[1:] *= 1000
    j_ij = j_ij.tolist()
    ex_params = {j_name: j[0] for j_name, j in zip(j_names, j_ij)}
    self.ex_params = ex_params
    return ex_params