from __future__ import annotations
import copy
import warnings
from typing import TYPE_CHECKING
from monty.dev import requires
from pymatgen.core.structure import IMolecule, Molecule
def make3d(self, forcefield: str='mmff94', steps: int=50) -> None:
    """
        A wrapper to pybel's make3D method generate a 3D structure from a
        2D or 0D structure.
        The 3D structure is made very quickly using a combination of rules
        (e.g. sp3 atoms should have four bonds arranged in a tetrahedron) and
        ring templates (e.g. cyclohexane is shaped like a chair). Once 3D
        coordinates are generated, hydrogens are added and a quick local
        optimization is carried out as default.

        The generated 3D structure can have clashes or have high energy
        structures due to some strain. Please consider to use the conformer
        search or geometry optimization to further optimize the structure.

        Args:
            forcefield: Default is mmff94. Options are 'gaff', 'ghemical',
                'mmff94', 'mmff94s', and 'uff'.
            steps: Default is 50.
        """
    pybelmol = pybel.Molecule(self._ob_mol)
    pybelmol.make3D(forcefield=forcefield, steps=steps)
    self._ob_mol = pybelmol.OBMol