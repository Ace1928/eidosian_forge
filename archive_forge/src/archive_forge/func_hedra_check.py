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
def hedra_check(dk: Tuple, ric: IC_Residue) -> None:
    """Confirm both hedra present for dihedron key, use default if set."""
    if dk[0:3] not in sbcic.hedra and dk[2::-1] not in sbcic.hedra:
        if defaults:
            default_hedron(dk[0:3], ric)
        else:
            print(f'{dk} missing h1')
    if dk[1:4] not in sbcic.hedra and dk[3:0:-1] not in sbcic.hedra:
        if defaults:
            default_hedron(dk[1:4], ric)
        else:
            print(f'{dk} missing h2')