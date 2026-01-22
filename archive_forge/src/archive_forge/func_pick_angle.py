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
def pick_angle(self, angle_key: Union[EKT, str]) -> Optional[Union['Hedron', 'Dihedron']]:
    """Get Hedron or Dihedron for angle_key.

        :param angle_key:
            - tuple of 3 or 4 AtomKeys
            - string of atom names ('CA') separated by :'s
            - string of [-1, 0, 1]<atom name> separated by ':'s. -1 is
              previous residue, 0 is this residue, 1 is next residue
            - psi, phi, omg, omega, chi1, chi2, chi3, chi4, chi5
            - tau (N-CA-C angle) see Richardson1981
            - tuples of AtomKeys is only access for alternate disordered atoms

        Observe that a residue's phi and omega dihedrals, as well as the hedra
        comprising them (including the N:Ca:C `tau` hedron), are stored in the
        n-1 di/hedra sets; this overlap is handled here, but may be an issue if
        accessing directly.

        The following print commands are equivalent (except for sidechains with
        non-carbon atoms for chi2)::

            ric = r.internal_coord
            print(
                r,
                ric.get_angle("psi"),
                ric.get_angle("phi"),
                ric.get_angle("omg"),
                ric.get_angle("tau"),
                ric.get_angle("chi2"),
            )
            print(
                r,
                ric.get_angle("N:CA:C:1N"),
                ric.get_angle("-1C:N:CA:C"),
                ric.get_angle("-1CA:-1C:N:CA"),
                ric.get_angle("N:CA:C"),
                ric.get_angle("CA:CB:CG:CD"),
            )

        See ic_data.py for detail of atoms in the enumerated sidechain angles
        and the backbone angles which do not span the peptide bond. Using 's'
        for current residue ('self') and 'n' for next residue, the spanning
        (overlapping) angles are::

                (sN, sCA, sC, nN)   # psi
                (sCA, sC, nN, nCA)  # omega i+1
                (sC, nN, nCA, nC)   # phi i+1
                (sCA, sC, nN)
                (sC, nN, nCA)
                (nN, nCA, nC)       # tau i+1

        :return: Matching Hedron, Dihedron, or None.
        """
    rval: Optional[Union['Hedron', 'Dihedron']] = None
    if isinstance(angle_key, tuple):
        rval = self._get_angle_for_tuple(angle_key)
        if rval is None and self.rprev:
            rval = self.rprev[0]._get_angle_for_tuple(angle_key)
    elif ':' in angle_key:
        angle_key = cast(EKT, self._get_ak_tuple(cast(str, angle_key)))
        if angle_key is None:
            return None
        rval = self._get_angle_for_tuple(angle_key)
        if rval is None and self.rprev:
            rval = self.rprev[0]._get_angle_for_tuple(angle_key)
    elif 'psi' == angle_key:
        if 0 == len(self.rnext):
            return None
        rn = self.rnext[0]
        sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
        nN = rn.rak('N')
        rval = self.dihedra.get((sN, sCA, sC, nN), None)
    elif 'phi' == angle_key:
        if 0 == len(self.rprev):
            return None
        rp = self.rprev[0]
        pC, sN, sCA = (rp.rak('C'), self.rak('N'), self.rak('CA'))
        sC = self.rak('C')
        rval = rp.dihedra.get((pC, sN, sCA, sC), None)
    elif 'omg' == angle_key or 'omega' == angle_key:
        if 0 == len(self.rprev):
            return None
        rp = self.rprev[0]
        pCA, pC, sN = (rp.rak('CA'), rp.rak('C'), self.rak('N'))
        sCA = self.rak('CA')
        rval = rp.dihedra.get((pCA, pC, sN, sCA), None)
    elif 'tau' == angle_key:
        sN, sCA, sC = (self.rak('N'), self.rak('CA'), self.rak('C'))
        rval = self.hedra.get((sN, sCA, sC), None)
        if rval is None and 0 != len(self.rprev):
            rp = self.rprev[0]
            rval = rp.hedra.get((sN, sCA, sC), None)
    elif angle_key.startswith('chi'):
        sclist = ic_data_sidechains.get(self.lc, None)
        if sclist is None:
            return None
        ndx = 2 * int(angle_key[-1]) - 1
        try:
            akl = sclist[ndx]
            if akl[4] == angle_key:
                klst = [self.rak(a) for a in akl[0:4]]
                tklst = cast(DKT, tuple(klst))
                rval = self.dihedra.get(tklst, None)
            else:
                return None
        except IndexError:
            return None
    return rval