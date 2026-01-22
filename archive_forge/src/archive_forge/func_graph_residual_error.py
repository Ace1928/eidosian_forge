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
def graph_residual_error(self) -> go.Figure:
    """Graphs the residual errors for all compounds after applying computed corrections."""
    if len(self.corrections) == 0:
        raise RuntimeError('Please call compute_corrections or compute_from_files to calculate corrections first')
    abs_errors = [abs(i) for i in self.diffs - np.dot(self.coeff_mat, self.corrections)]
    labels_graph = self.names.copy()
    abs_errors, labels_graph = (list(t) for t in zip(*sorted(zip(abs_errors, labels_graph))))
    n_err = len(abs_errors)
    fig = go.Figure(data=go.Scatter(x=np.linspace(1, n_err, n_err), y=abs_errors, mode='markers', text=labels_graph), layout=dict(title='Residual Errors', yaxis=dict(title='Residual Error (eV/atom)')))
    print('Residual Error:')
    print(f'Median = {np.median(abs_errors)}')
    print(f'Mean = {np.mean(abs_errors)}')
    print(f'Std Dev = {np.std(abs_errors)}')
    print('Original Error:')
    print(f'Median = {abs(np.median(self.diffs))}')
    print(f'Mean = {abs(np.mean(self.diffs))}')
    print(f'Std Dev = {np.std(self.diffs)}')
    return fig