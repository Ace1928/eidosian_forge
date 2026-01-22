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
def get_structure_by_formula(self, formula: str, **kwargs) -> list[dict[str, str | int | Structure]]:
    """Queries the COD for structures by formula. Requires mysql executable to
        be in the path.

        Args:
            formula (str): Chemical formula.
            kwargs: All kwargs supported by Structure.from_str.

        Returns:
            A list of dict of the format [{"structure": Structure, "cod_id": int, "sg": "P n m a"}]
        """
    structures: list[dict[str, str | int | Structure]] = []
    sql = f'select file, sg from data where formula="- {Composition(formula).hill_formula} -"'
    text = self.query(sql).split('\n')
    text.pop(0)
    for line in text:
        if line.strip():
            cod_id, sg = line.split('\t')
            response = requests.get(f'http://www.crystallography.net/cod/{cod_id.strip()}.cif')
            try:
                struct = Structure.from_str(response.text, fmt='cif', **kwargs)
                structures.append({'structure': struct, 'cod_id': int(cod_id), 'sg': sg})
            except Exception:
                warnings.warn(f'\nStructure.from_str failed while parsing CIF file:\n{response.text}')
                raise
    return structures