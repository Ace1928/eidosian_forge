from Bio.PDB.Entity import Entity
def atom_to_internal_coordinates(self, verbose: bool=False) -> None:
    """Create/update internal coordinates from Atom X,Y,Z coordinates.

        Internal coordinates are bond length, angle and dihedral angles.

        :param verbose bool: default False
            describe runtime problems

        """
    for chn in self.get_chains():
        chn.atom_to_internal_coordinates(verbose)