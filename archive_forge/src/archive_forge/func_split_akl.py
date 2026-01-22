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
def split_akl(self, lst: Union[Tuple['AtomKey', ...], List['AtomKey']], missingOK: bool=False) -> List[Tuple['AtomKey', ...]]:
    """Get AtomKeys for this residue (ak_set) for generic list of AtomKeys.

        Changes and/or expands a list of 'generic' AtomKeys (e.g. 'N, C, C') to
        be specific to this Residue's altlocs etc., e.g.
        '(N-Ca_A_0.3-C, N-Ca_B_0.7-C)'

        Given a list of AtomKeys for a Hedron or Dihedron,
          return:
                list of matching atomkeys that have id3_dh in this residue
                (ak may change if occupancy != 1.00)

            or
                multiple lists of matching atomkeys expanded for all atom altlocs

            or
                empty list if any of atom_coord(ak) missing and not missingOK

        :param list lst: list[3] or [4] of AtomKeys.
            Non-altloc AtomKeys to match to specific AtomKeys for this residue
        :param bool missingOK: default False, see above.
        """
    altloc_ndx = AtomKey.fields.altloc
    occ_ndx = AtomKey.fields.occ
    edraLst: List[Tuple[AtomKey, ...]] = []
    altlocs = set()
    posnAltlocs: Dict['AtomKey', Set[str]] = {}
    akMap = {}
    for ak in lst:
        posnAltlocs[ak] = set()
        if ak in self.ak_set and ak.akl[altloc_ndx] is None and (ak.akl[occ_ndx] is None):
            edraLst.append((ak,))
        else:
            ak2_lst = []
            for ak2 in self.ak_set:
                if ak.altloc_match(ak2):
                    ak2_lst.append(ak2)
                    akMap[ak2] = ak
                    altloc = ak2.akl[altloc_ndx]
                    if altloc is not None:
                        altlocs.add(altloc)
                        posnAltlocs[ak].add(altloc)
            edraLst.append(tuple(ak2_lst))
    maxc = 0
    for akl in edraLst:
        lenAKL = len(akl)
        if 0 == lenAKL and (not missingOK):
            return []
        elif maxc < lenAKL:
            maxc = lenAKL
    if 1 == maxc:
        newAKL = []
        for akl in edraLst:
            if akl:
                newAKL.append(akl[0])
        return [tuple(newAKL)]
    else:
        new_edraLst = []
        for al in altlocs:
            alhl = []
            for akl in edraLst:
                lenAKL = len(akl)
                if 0 == lenAKL:
                    continue
                if 1 == lenAKL:
                    alhl.append(akl[0])
                elif al not in posnAltlocs[akMap[akl[0]]]:
                    alhl.append(sorted(akl)[0])
                else:
                    for ak in akl:
                        if ak.akl[altloc_ndx] == al:
                            alhl.append(ak)
            new_edraLst.append(tuple(alhl))
        return new_edraLst