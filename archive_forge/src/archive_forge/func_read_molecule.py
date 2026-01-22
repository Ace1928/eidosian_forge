from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.core import Molecule
from pymatgen.io.core import InputFile
from .utils import lower_and_check_unique, read_pattern, read_table_pattern
@staticmethod
def read_molecule(string: str) -> Molecule | list[Molecule] | Literal['read']:
    """
        Read molecule from string.

        Args:
            string (str): String

        Returns:
            Molecule
        """
    charge = spin_mult = None
    patterns = {'read': '^\\s*\\$molecule\\n\\s*(read)', 'charge': '^\\s*\\$molecule\\n\\s*((?:\\-)*\\d+)\\s+\\d+', 'spin_mult': '^\\s*\\$molecule\\n\\s(?:\\-)*\\d+\\s*((?:\\-)*\\d+)', 'fragment': '^\\s*\\$molecule\\n\\s*(?:\\-)*\\d+\\s+\\d+\\s*\\n\\s*(\\-\\-)'}
    matches = read_pattern(string, patterns)
    if 'read' in matches:
        return 'read'
    if 'charge' in matches:
        charge = float(matches['charge'][0][0])
    if 'spin_mult' in matches:
        spin_mult = int(matches['spin_mult'][0][0])
    multi_mol = 'fragment' in matches
    if not multi_mol:
        header = '^\\s*\\$molecule\\n\\s*(?:\\-)*\\d+\\s+(?:\\-)*\\d+'
        row = '\\s*([A-Za-z]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
        footer = '^\\$end'
        mol_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
        species = [val[0] for val in mol_table[0]]
        coords = [[float(val[1]), float(val[2]), float(val[3])] for val in mol_table[0]]
        if charge is None:
            mol = Molecule(species=species, coords=coords)
        else:
            mol = Molecule(species=species, coords=coords, charge=charge, spin_multiplicity=spin_mult)
        return mol
    header = '\\s*(?:\\-)*\\d+\\s+(?:\\-)*\\d+'
    row = '\\s*([A-Za-z]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
    footer = '(:?(:?\\-\\-)|(:?\\$end))'
    molecules = []
    patterns = {'charge_spin': '\\s*\\-\\-\\s*([\\-0-9]+)\\s+([\\-0-9]+)'}
    matches = read_pattern(string, patterns)
    mol_table = read_table_pattern(string, header_pattern=header, row_pattern=row, footer_pattern=footer)
    for match, table in zip(matches.get('charge_spin'), mol_table):
        charge = int(match[0])
        spin = int(match[1])
        species = [val[0] for val in table]
        coords = [[float(val[1]), float(val[2]), float(val[3])] for val in table]
        mol = Molecule(species=species, coords=coords, charge=charge, spin_multiplicity=spin)
        molecules.append(mol)
    return molecules