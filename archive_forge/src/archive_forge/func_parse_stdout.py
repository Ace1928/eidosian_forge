from __future__ import annotations
import logging
import subprocess
from shutil import which
import pandas as pd
from monty.dev import requires
from monty.json import MSONable
from pymatgen.analysis.magnetism.heisenberg import HeisenbergMapper
@staticmethod
def parse_stdout(vamp_stdout, n_mats: int) -> tuple:
    """Parse stdout from Vampire.

        Args:
            vamp_stdout (txt file): Vampire 'output' file.
            n_mats (int): Number of materials in Vampire simulation.

        Returns:
            parsed_out (DataFrame): MSONable vampire output.
            critical_temp (float): Calculated critical temp.
        """
    names = ['T', 'm_total', *[f'm_{idx + 1}' for idx in range(n_mats)], 'X_x', 'X_y', 'X_z', 'X_m', 'nan']
    df = pd.read_csv(vamp_stdout, sep='\t', skiprows=9, header=None, names=names)
    df = df.drop('nan', axis=1)
    parsed_out = df.to_json()
    critical_temp = df.iloc[df.X_m.idxmax()]['T']
    return (parsed_out, critical_temp)