from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def svp_template(svp: dict) -> str:
    """
        Template for the $svp section.

        Args:
            svp: dict of SVP parameters, e.g.
            {"rhoiso": "0.001", "nptleb": "1202", "itrngr": "2", "irotgr": "2"}

        Returns:
            str: the $svp section. Note that all parameters will be concatenated onto
                a single line formatted as a FORTRAN namelist. This is necessary
                because the isodensity SS(V)PE model in Q-Chem calls a secondary code.
        """
    svp_list = []
    svp_list.append('$svp')
    param_list = [f'{_key}={value}' for _key, value in svp.items()]
    svp_list.extend((', '.join(param_list), '$end'))
    return '\n'.join(svp_list)