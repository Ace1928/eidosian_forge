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
def init_atom_coords(self) -> None:
    """Set chain level di/hedra initial coords from angles and distances.

        Initializes atom coordinates in local coordinate space for hedra and
        dihedra, will be transformed appropriately later by :data:`dCoordSpace`
        matrices for assembly.
        """
    if not np.all(self.dAtoms_needs_update):
        self.dAtoms_needs_update |= self.hAtoms_needs_update[self.dH1ndx] | self.hAtoms_needs_update[self.dH2ndx]
        self.dcsValid &= np.logical_not(self.dAtoms_needs_update)
    mdFwd = self.dFwd & self.dAtoms_needs_update
    mdRev = self.dRev & self.dAtoms_needs_update
    udFwd = self.dFwd[self.dAtoms_needs_update]
    udRev = self.dRev[self.dAtoms_needs_update]
    '\n        if dbg:\n            print("mdFwd", mdFwd[0:10])\n            print("mdRev", mdRev[0:10])\n            print("udFwd", udFwd[0:10])\n            print("udRev", udRev[0:10])\n        '
    if np.any(self.hAtoms_needs_update):
        sar = np.deg2rad(180.0 - self.hedraAngle[self.hAtoms_needs_update])
        sinSar = np.sin(sar)
        cosSarN = np.cos(sar) * -1
        '\n            if dbg:\n                print("sar", sar[0:10])\n            '
        self.hAtoms[:, 2, 2][self.hAtoms_needs_update] = self.hedraL23[self.hAtoms_needs_update]
        self.hAtoms[:, 0, 0][self.hAtoms_needs_update] = sinSar * self.hedraL12[self.hAtoms_needs_update]
        self.hAtoms[:, 0, 2][self.hAtoms_needs_update] = cosSarN * self.hedraL12[self.hAtoms_needs_update]
        '\n            if dbg:\n                print("hAtoms_needs_update", self.hAtoms_needs_update[0:10])\n                print("self.hAtoms", self.hAtoms[0:10])\n            '
        self.hAtomsR[:, 0, 2][self.hAtoms_needs_update] = self.hedraL12[self.hAtoms_needs_update]
        self.hAtomsR[:, 2, 0][self.hAtoms_needs_update] = sinSar * self.hedraL23[self.hAtoms_needs_update]
        self.hAtomsR[:, 2, 2][self.hAtoms_needs_update] = cosSarN * self.hedraL23[self.hAtoms_needs_update]
        '\n            if dbg:\n                print("self.hAtomsR", self.hAtomsR[0:10])\n            '
        self.hAtoms_needs_update[...] = False
        dhlen = np.sum(self.dAtoms_needs_update)
        self.a4_pre_rotation[mdRev] = self.hAtoms[self.dH2ndx, 0][mdRev]
        self.a4_pre_rotation[mdFwd] = self.hAtomsR[self.dH2ndx, 2][mdFwd]
        self.a4_pre_rotation[:, 2][self.dAtoms_needs_update] = np.multiply(self.a4_pre_rotation[:, 2][self.dAtoms_needs_update], -1)
        a4shift = np.empty(dhlen)
        a4shift[udRev] = self.hedraL23[self.dH2ndx][mdRev]
        a4shift[udFwd] = self.hedraL12[self.dH2ndx][mdFwd]
        self.a4_pre_rotation[:, 2][self.dAtoms_needs_update] = np.add(self.a4_pre_rotation[:, 2][self.dAtoms_needs_update], a4shift)
        '\n            if dbg:\n                print("dhlen", dhlen)\n                print("a4shift", a4shift[0:10])\n                print("a4_pre_rotation", self.a4_pre_rotation[0:10])\n            '
        dH1atoms = self.hAtoms[self.dH1ndx]
        dH1atomsR = self.hAtomsR[self.dH1ndx]
        self.dAtoms[:, :3][mdFwd] = dH1atoms[mdFwd]
        self.dAtoms[:, :3][mdRev] = dH1atomsR[:, 2::-1][mdRev]
        '\n            if dbg:\n                print("dH1atoms", dH1atoms[0:10])\n                print("dH1atosR", dH1atomsR[0:10])\n                print("dAtoms", self.dAtoms[0:10])\n            '
    '\n        if dbg:\n            print("dangle-rads", self.dihedraAngleRads[0:10])\n        '
    rz = multi_rot_Z(self.dihedraAngleRads[self.dAtoms_needs_update])
    a4rot = np.matmul(rz, self.a4_pre_rotation[self.dAtoms_needs_update][:].reshape(-1, 4, 1)).reshape(-1, 4)
    self.dAtoms[:, 3][mdFwd] = a4rot[udFwd]
    self.dAtoms[:, 3][mdRev] = a4rot[udRev]
    '\n        if dbg:\n            print("rz", rz[0:3])\n            print("dAtoms", self.dAtoms[0:10])\n        '
    self.dAtoms_needs_update[...] = False
    '\n        if dbg:\n            print("initNCaCs", self.initNCaCs)\n        '
    for iNCaC in self.initNCaCs:
        invalid = True
        if np.all(self.atomArrayValid[[self.atomArrayIndex[ak] for ak in iNCaC]]):
            invalid = False
        if invalid:
            hatoms = self.hAtoms[self.hedraNdx[iNCaC]]
            for i in range(3):
                andx = self.atomArrayIndex[iNCaC[i]]
                self.atomArray[andx] = hatoms[i]
                self.atomArrayValid[andx] = True
        '\n            if dbg:\n                hatoms = self.hAtoms[self.hedraNdx[iNCaC]]\n                print("hedraNdx iNCaC", self.hedraNdx[iNCaC])\n                print("hatoms", hatoms)\n            '