import re
from collections import deque, namedtuple
import copy
from numbers import Integral
import numpy as np  # type: ignore
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB.vectors import multi_coord_space, multi_rot_Z
from Bio.PDB.vectors import coord_space
from Bio.PDB.ic_data import ic_data_backbone, ic_data_sidechains
from Bio.PDB.ic_data import primary_angles
from Bio.PDB.ic_data import ic_data_sidechain_extras, residue_atom_bond_state
from Bio.PDB.ic_data import dihedra_primary_defaults, hedra_defaults
from typing import (
def build_atomArray(self) -> None:
    """Build :class:`IC_Chain` numpy coordinate array from biopython atoms.

        See also :meth:`.init_edra` for more complete initialization of IC_Chain.

        Inputs:
            self.akset : set
                :class:`AtomKey` s in this chain

        Generates:
            self.AAsiz : int
                number of atoms in chain (len(akset))
            self.aktuple : AAsiz x AtomKeys
                sorted akset AtomKeys
            self.atomArrayIndex : [AAsiz] of int
                numerical index for each AtomKey in aktuple
            self.atomArrayValid : AAsiz x bool
                atomArray coordinates current with internal coordinates if True
            self.atomArray : AAsiz x np.float64[4]
                homogeneous atom coordinates; Biopython :class:`.Atom`
                coordinates are view into this array after execution
            rak_cache : dict
                lookup cache for AtomKeys for each residue

        """

    def setAtom(res, atm):
        ak = AtomKey(res.internal_coord, atm)
        try:
            ndx = self.atomArrayIndex[ak]
        except KeyError:
            return
        self.atomArray[ndx, 0:3] = atm.coord
        atm.coord = self.atomArray[ndx, 0:3]
        self.atomArrayValid[ndx] = True
        self.bpAtomArray[ndx] = atm

    def setResAtms(res):
        for atm in res.get_atoms():
            if atm.is_disordered():
                if IC_Residue.no_altloc:
                    setAtom(res, atm.selected_child)
                else:
                    for altAtom in atm.child_dict.values():
                        setAtom(res, altAtom)
            else:
                setAtom(res, atm)
    self.AAsiz = len(self.akset)
    self.aktuple = tuple(sorted(self.akset))
    self.atomArrayIndex = dict(zip(self.aktuple, range(self.AAsiz)))
    self.atomArrayValid = np.zeros(self.AAsiz, dtype=bool)
    self.atomArray = np.zeros((self.AAsiz, 4), dtype=np.float64)
    self.atomArray[:, 3] = 1.0
    self.bpAtomArray = [None] * self.AAsiz
    for ric in self.ordered_aa_ic_list:
        setResAtms(ric.residue)
        if ric.akc == {}:
            ric._build_rak_cache()