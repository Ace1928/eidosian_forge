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
def compare_residues(e0: Union[Structure, Model, Chain], e1: Union[Structure, Model, Chain], verbose: bool=False, quick: bool=False, rtol: float=None, atol: float=None) -> Dict[str, Any]:
    """Compare full IDs and atom coordinates for 2 Biopython PDB entities.

    Skip DNA and HETATMs.

    :param Entity e0,e1: Biopython PDB Entity objects (S, M or C).
        Structures, Models or Chains to be compared
    :param bool verbose:
        Whether to print mismatch info, default False
    :param bool quick: default False.
        Only check atomArrays are identical, aCoordMatchCount=0 if different
    :param float rtol, atol: default 1e-03, 1e-03 or round to 3 places.
        NumPy allclose parameters; default is to round atom coordinates to 3
        places and test equal.  For 'quick' will use defaults above for
        comparing atomArrays
    :returns dict:
        Result counts for Residues, Full ID match Residues, Atoms,
        Full ID match atoms, and Coordinate match atoms; report string;
        error status (bool)
    """
    cmpdict: Dict[str, Any] = {}
    cmpdict['chains'] = []
    cmpdict['residues'] = 0
    cmpdict['rCount'] = 0
    cmpdict['rMatchCount'] = 0
    cmpdict['rpnMismatchCount'] = 0
    cmpdict['aCount'] = 0
    cmpdict['disAtmCount'] = 0
    cmpdict['aCoordMatchCount'] = 0
    cmpdict['aFullIdMatchCount'] = 0
    cmpdict['id0'] = e0.get_full_id()
    cmpdict['id1'] = e1.get_full_id()
    cmpdict['pass'] = None
    cmpdict['report'] = None
    if quick:
        if isinstance(e0, Chain):
            if e0.internal_coord.atomArray is not None and np.shape(e0.internal_coord.atomArray) == np.shape(e1.internal_coord.atomArray) and np.allclose(e0.internal_coord.atomArray, e1.internal_coord.atomArray, rtol=0.001 if rtol is None else rtol, atol=0.001 if atol is None else atol):
                cmpdict['aCount'] = np.size(e0.internal_coord.atomArray, 0)
                cmpdict['aCoordMatchCount'] = np.size(e0.internal_coord.atomArray, 0)
                if cmpdict['aCoordMatchCount'] > 0:
                    cmpdict['pass'] = True
                else:
                    cmpdict['pass'] = False
            else:
                cmpdict['aCount'] = 0 if e0.internal_coord.atomArray is None else np.size(e0.internal_coord.atomArray, 0)
                cmpdict['pass'] = False
        else:
            cmpdict['pass'] = True
            for c0, c1 in zip_longest(e0.get_chains(), e1.get_chains()):
                if c0.internal_coord.atomArray is not None:
                    if np.allclose(c0.internal_coord.atomArray, c1.internal_coord.atomArray, rtol=0.001 if rtol is None else rtol, atol=0.001 if atol is None else atol):
                        cmpdict['aCoordMatchCount'] += np.size(c0.internal_coord.atomArray, 0)
                    else:
                        cmpdict['pass'] = False
                    cmpdict['aCount'] += np.size(c0.internal_coord.atomArray, 0)
            if cmpdict['aCoordMatchCount'] < cmpdict['aCount']:
                cmpdict['pass'] = False
    else:
        for r0, r1 in zip_longest(e0.get_residues(), e1.get_residues()):
            if 2 == r0.is_disordered() == r1.is_disordered():
                for dr0, dr1 in zip_longest(r0.child_dict.values(), r1.child_dict.values()):
                    _cmp_res(dr0, dr1, verbose, cmpdict, rtol=rtol, atol=atol)
            else:
                _cmp_res(r0, r1, verbose, cmpdict, rtol=rtol, atol=atol)
        if cmpdict['rMatchCount'] == cmpdict['rCount'] and cmpdict['aCoordMatchCount'] == cmpdict['aCount'] and (cmpdict['aFullIdMatchCount'] == cmpdict['aCount']) and (cmpdict['rpnMismatchCount'] == 0):
            cmpdict['pass'] = True
        else:
            cmpdict['pass'] = False
    rstr = '{}:{} {} -- {} of {} residue IDs match; {} residues {} atom coords, {} full IDs of {} atoms ({} disordered) match : {}'.format(cmpdict['id0'], cmpdict['id1'], cmpdict['chains'], cmpdict['rMatchCount'], cmpdict['rCount'], cmpdict['residues'], cmpdict['aCoordMatchCount'], cmpdict['aFullIdMatchCount'], cmpdict['aCount'], cmpdict['disAtmCount'], 'ERROR' if not cmpdict['pass'] else 'ALL OK')
    if not cmpdict['pass']:
        if cmpdict['rMatchCount'] != cmpdict['rCount']:
            rstr += ' -RESIDUE IDS-'
        if cmpdict['aCoordMatchCount'] != cmpdict['aFullIdMatchCount']:
            rstr += ' -COORDINATES-'
        if cmpdict['aFullIdMatchCount'] != cmpdict['aCount']:
            rstr += ' -ATOM IDS-'
    cmpdict['report'] = rstr
    return cmpdict