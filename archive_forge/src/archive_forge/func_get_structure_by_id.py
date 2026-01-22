from __future__ import annotations
import re
import subprocess
import warnings
from shutil import which
import requests
from monty.dev import requires
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
def get_structure_by_id(self, cod_id, **kwargs):
    """Queries the COD for a structure by id.

        Args:
            cod_id (int): COD id.
            kwargs: All kwargs supported by Structure.from_str.

        Returns:
            A Structure.
        """
    response = requests.get(f'http://{self.url}/cod/{cod_id}.cif')
    return Structure.from_str(response.text, fmt='cif', **kwargs)