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
def build_edraArrays(self) -> None:
    """Build chain level hedra and dihedra arrays.

        Used by :meth:`init_edra` and :meth:`_hedraDict2chain`.  Should be
        private method but exposed for documentation.

        Inputs:
            self.dihedraLen : int
                number of dihedra needed
            self.hedraLen : int
                number of hedra needed
            self.AAsiz : int
                length of atomArray
            self.hedraNdx : dict
                maps hedron keys to range(hedraLen)
            self.dihedraNdx : dict
                maps dihedron keys to range(dihedraLen)
            self.hedra : dict
                maps Hedra keys to Hedra for chain
            self.atomArray : AAsiz x np.float64[4]
                homogeneous atom coordinates for chain
            self.atomArrayIndex : dict
                maps AtomKeys to atomArray
            self.atomArrayValid : AAsiz x bool
                indicates coord is up-to-date

        Generates:
            self.dCoordSpace : [2][dihedraLen][4][4]
                transforms to/from dihedron coordinate space
            self.dcsValid : dihedraLen x bool
                indicates dCoordSpace is current
            self.hAtoms : hedraLen x 3 x np.float64[4]
                atom coordinates in hCoordSpace
            self.hAtomsR : hedraLen x 3 x np.float64[4]
                hAtoms in reverse order (trading space for time)
            self.hAtoms_needs_update : hedraLen x bool
                indicates hAtoms, hAtoms current
            self.a2h_map : AAsiz x [int ...]
                maps atomArrayIndex to hedraNdx's with that atom
            self.a2ha_map : [hedraLen x 3]
                AtomNdx's in hedraNdx order
            self.h2aa : hedraLen x [int ...]
                maps hedraNdx to atomNdx's in hedron (reshaped later)
            Hedron.ndx : int
                self.hedraNdx value stored inside Hedron object
            self.dRev : dihedraLen x bool
                dihedron reversed if true
            self.dH1ndx, dH2ndx : [dihedraLen]
                hedraNdx's for 1st and 2nd hedra
            self.h1d_map : hedraLen x []
                hedraNdx -> [dihedra using hedron]
            Dihedron.h1key, h2key : [AtomKey ...]
                hedron keys for dihedron, reversed as needed
            Dihedron.hedron1, hedron2 : Hedron
                references inside dihedron to hedra
            Dihedron.ndx : int
                self.dihedraNdx info inside Dihedron object
            Dihedron.cst, rcst : np.float64p4][4]
                dCoordSpace references inside Dihedron
            self.a2da_map : [dihedraLen x 4]
                AtomNdx's in dihedraNdx order
            self.d2a_map : [dihedraLen x [4]]
                AtomNdx's for each dihedron (reshaped a2da_map)
            self.dFwd : bool
                dihedron is not Reversed if True
            self.a2d_map : AAsiz x [[dihedraNdx]
                [atom ndx 0-3 of atom in dihedron]], maps atom indexes to
                dihedra and atoms in them
            self.dAtoms_needs_update : dihedraLen x bool
                atoms in h1, h2 are current if False

        """
    self.dCoordSpace: np.ndarray = np.empty((2, self.dihedraLen, 4, 4), dtype=np.float64)
    self.dcsValid: np.ndarray = np.zeros(self.dihedraLen, dtype=bool)
    self.hAtoms: np.ndarray = np.zeros((self.hedraLen, 3, 4), dtype=np.float64)
    self.hAtoms[:, :, 3] = 1.0
    self.hAtomsR: np.ndarray = np.copy(self.hAtoms)
    self.hAtoms_needs_update = np.full(self.hedraLen, True)
    a2ha_map = {}
    self.a2h_map = [[] for _ in range(self.AAsiz)]
    h2aa = [[] for _ in range(self.hedraLen)]
    for hk, hndx in self.hedraNdx.items():
        hstep = hndx * 3
        for i in range(3):
            ndx = self.atomArrayIndex[hk[i]]
            a2ha_map[hstep + i] = ndx
        self.hedra[hk].ndx = hndx
        for ak in self.hedra[hk].atomkeys:
            akndx = self.atomArrayIndex[ak]
            h2aa[hndx].append(akndx)
            self.a2h_map[akndx].append(hndx)
    self.a2ha_map = np.array(tuple(a2ha_map.values()))
    self.h2aa = np.array(h2aa)
    self.dAtoms: np.ndarray = np.empty((self.dihedraLen, 4, 4), dtype=np.float64)
    self.dAtoms[:, :, 3] = 1.0
    self.a4_pre_rotation = np.empty((self.dihedraLen, 4))
    a2da_map = {}
    a2d_map = [[[], []] for _ in range(self.AAsiz)]
    self.dRev: np.ndarray = np.zeros(self.dihedraLen, dtype=bool)
    self.dH1ndx = np.empty(self.dihedraLen, dtype=np.int64)
    self.dH2ndx = np.empty(self.dihedraLen, dtype=np.int64)
    self.h1d_map = [[] for _ in range(self.hedraLen)]
    self.id3_dh_index = {k[0:3]: [] for k in self.dihedraNdx.keys()}
    self.id32_dh_index = {k[1:4]: [] for k in self.dihedraNdx.keys()}
    for dk, dndx in self.dihedraNdx.items():
        dstep = dndx * 4
        did3 = dk[0:3]
        did32 = dk[1:4]
        d = self.dihedra[dk]
        for i in range(4):
            ndx = self.atomArrayIndex[dk[i]]
            a2da_map[dstep + i] = ndx
            a2d_map[ndx][0].append(dndx)
            a2d_map[ndx][1].append(i)
        try:
            d.h1key = did3
            d.h2key = did32
            h1ndx = self.hedraNdx[d.h1key]
        except KeyError:
            d.h1key = dk[2::-1]
            d.h2key = dk[3:0:-1]
            h1ndx = self.hedraNdx[d.h1key]
            self.dRev[dndx] = True
            d.reverse = True
        h2ndx = self.hedraNdx[d.h2key]
        d.hedron1 = self.hedra[d.h1key]
        d.hedron2 = self.hedra[d.h2key]
        self.dH1ndx[dndx] = h1ndx
        self.dH2ndx[dndx] = h2ndx
        self.h1d_map[h1ndx].append(dndx)
        d.ndx = dndx
        d.cst = self.dCoordSpace[0][dndx]
        d.rcst = self.dCoordSpace[1][dndx]
        self.id3_dh_index[did3].append(dk)
        self.id32_dh_index[did32].append(dk)
    self.a2da_map = np.array(tuple(a2da_map.values()))
    self.d2a_map = self.a2da_map.reshape(-1, 4)
    self.dFwd = self.dRev != True
    self.a2d_map = [(np.array(xi[0]), np.array(xi[1])) for xi in a2d_map]
    self.dAtoms_needs_update = np.full(self.dihedraLen, True)