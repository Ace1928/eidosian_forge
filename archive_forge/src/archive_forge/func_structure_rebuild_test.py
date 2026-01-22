import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
def structure_rebuild_test(entity, verbose: bool=False, quick: bool=False) -> Dict:
    """Test rebuild PDB structure from internal coordinates.

    Generates internal coordinates for entity and writes to a .pic file in
    memory, then generates XYZ coordinates from the .pic file and compares the
    resulting entity against the original.

    See :data:`IC_Residue.pic_accuracy` to vary numeric accuracy of the
    intermediate .pic file if the only issue is small differences in coordinates.

    Note that with default settings, deuterated initial structures will fail
    the comparison, as will structures loaded with alternate `IC_Residue.accept_atoms`
    settings.  Use `quick=True` and/or variations on `AtomKey.d2h` and
    `IC_Residue.accept_atoms` settings.

    :param Entity entity: Biopython Structure, Model or Chain.
        Structure to test
    :param bool verbose: default False.
        print extra messages
    :param bool quick: default False.
        only check the internal coords atomArrays are identical
    :returns: dict
        comparison dict from :func:`.compare_residues`
    """
    sp = StringIO()
    entity.atom_to_internal_coordinates(verbose)
    write_PIC(entity, sp)
    sp.seek(0)
    pdb2 = read_PIC(sp, verbose=verbose, quick=quick)
    if isinstance(entity, Chain):
        pdb2 = next(pdb2.get_chains())
    if verbose:
        report_IC(pdb2, verbose=True)
    pdb2.internal_to_atom_coordinates(verbose)
    r = compare_residues(entity, pdb2, verbose=verbose, quick=quick)
    return r