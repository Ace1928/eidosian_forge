from __future__ import annotations
import re
import subprocess
import warnings
from shutil import which
import requests
from monty.dev import requires
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
@requires(which('mysql'), 'mysql must be installed to use this query.')
def get_cod_ids(self, formula):
    """Queries the COD for all cod ids associated with a formula. Requires
        mysql executable to be in the path.

        Args:
            formula (str): Formula.

        Returns:
            List of cod ids.
        """
    cod_formula = Composition(formula).hill_formula
    sql = f'select file from data where formula="- {cod_formula} -"'
    text = self.query(sql).split('\n')
    cod_ids = []
    for line in text:
        match = re.search('(\\d+)', line)
        if match:
            cod_ids.append(int(match.group(1)))
    return cod_ids