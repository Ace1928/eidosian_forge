from __future__ import annotations
import os
import warnings
import numpy as np
import plotly.graph_objects as go
from monty.serialization import loadfn
from ruamel import yaml
from scipy.optimize import curve_fit
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.structure_analyzer import sulfide_type
from pymatgen.core import Composition, Element
def graph_residual_error_per_species(self, specie: str) -> go.Figure:
    """Graphs the residual errors for each compound that contains specie after applying computed corrections.

        Args:
            specie: the specie/group that residual errors are being plotted for

        Raises:
            ValueError: the specie is not a valid specie that this class fits corrections for
        """
    if specie not in self.species:
        raise ValueError('not a valid specie')
    if len(self.corrections) == 0:
        raise RuntimeError('Please call compute_corrections or compute_from_files to calculate corrections first')
    abs_errors = [abs(i) for i in self.diffs - np.dot(self.coeff_mat, self.corrections)]
    labels_species = self.names.copy()
    diffs_cpy = self.diffs.copy()
    n_species = len(labels_species)
    if specie in ('oxide', 'peroxide', 'superoxide', 'S'):
        if specie == 'oxide':
            compounds = self.oxides
        elif specie == 'peroxide':
            compounds = self.peroxides
        elif specie == 'superoxides':
            compounds = self.superoxides
        else:
            compounds = self.sulfides
        for idx in range(n_species):
            if labels_species[n_species - idx - 1] not in compounds:
                del labels_species[n_species - idx - 1]
                del abs_errors[n_species - idx - 1]
                del diffs_cpy[n_species - idx - 1]
    else:
        for idx in range(n_species):
            if not Composition(labels_species[n_species - idx - 1])[specie]:
                del labels_species[n_species - idx - 1]
                del abs_errors[n_species - idx - 1]
                del diffs_cpy[n_species - idx - 1]
    abs_errors, labels_species = (list(tup) for tup in zip(*sorted(zip(abs_errors, labels_species))))
    n_err = len(abs_errors)
    fig = go.Figure(data=go.Scatter(x=np.linspace(1, n_err, n_err), y=abs_errors, mode='markers', text=labels_species), layout=dict(title=dict(text=f'Residual Errors for {specie}'), yaxis=dict(title='Residual Error (eV/atom)')))
    print('Residual Error:')
    print(f'Median = {np.median(np.array(abs_errors))}')
    print(f'Mean = {np.mean(np.array(abs_errors))}')
    print(f'Std Dev = {np.std(np.array(abs_errors))}')
    print('Original Error:')
    print(f'Median = {abs(np.median(np.array(diffs_cpy)))}')
    print(f'Mean = {abs(np.mean(np.array(diffs_cpy)))}')
    print(f'Std Dev = {np.std(np.array(diffs_cpy))}')
    return fig