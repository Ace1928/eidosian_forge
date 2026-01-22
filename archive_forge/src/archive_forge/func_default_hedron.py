import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def default_hedron(ek: Tuple, ric: IC_Residue) -> None:
    """Create Hedron based on same re_class hedra in ref database.

        Adds Hedron to current Chain.internal_coord, see ic_data for default
        values and reference database source.
        """
    atomkeys = []
    hkey = None
    atmNdx = AtomKey.fields.atm
    resNdx = AtomKey.fields.resname
    resPos = AtomKey.fields.respos
    atomkeys = [ek[i].akl for i in range(3)]
    atpl = tuple([atomkeys[i][atmNdx] for i in range(3)])
    res = atomkeys[0][resNdx]
    if atomkeys[0][resPos] != atomkeys[2][resPos] or atpl == ('N', 'CA', 'C') or atpl in ic_data_backbone or (res not in ['A', 'G'] and atpl in ic_data_sidechains[res]):
        hkey = ek
        rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(3)]
        try:
            dflts = hedra_defaults[''.join(rhcl)][0]
        except KeyError:
            if atomkeys[0][resPos] == atomkeys[1][resPos]:
                rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(2)]
                rhc = ''.join(rhcl) + 'X' + atomkeys[2][atmNdx]
            else:
                rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(1, 3)]
                rhc = 'X' + atomkeys[0][atmNdx] + ''.join(rhcl)
            dflts = hedra_defaults[rhc][0]
    else:
        hkey = ek[::-1]
        rhcl = [atomkeys[i][resNdx] + atomkeys[i][atmNdx] for i in range(2, -1, -1)]
        dflts = hedra_defaults[''.join(rhcl)][0]
    process_hedron(str(hkey[0]), str(hkey[1]), str(hkey[2]), dflts[0], dflts[1], dflts[2], ric)
    if verbose:
        print(f' default for {ek}')