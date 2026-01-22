from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
def visualize_directionality_quality_criterion(self, other: ThermalDisplacementMatrices, filename: str | PathLike='visualization.vesta', which_structure: int=0) -> None:
    """Will create a VESTA file for visualization of the directionality criterion.

        Args:
            other: ThermalDisplacementMatrices
            filename:           Filename of the VESTA file
            which_structure:    0 means structure of the self object will be used, 1 means structure of the other
                                object will be used
        """
    result = self.compute_directionality_quality_criterion(other=other)
    matrix_cif = self.thermal_displacement_matrix_cif if self.thermal_displacement_matrix_cif is not None else self.get_reduced_matrix(self.Ucif)
    if which_structure == 0:
        structure = self.structure
    elif which_structure == 1:
        structure = other.structure
    with open(filename, mode='w') as file:
        file.write('#VESTA_FORMAT_VERSION 3.5.4\n \n \n')
        file.write('CRYSTAL\n\n')
        file.write('TITLE\n')
        file.write('Directionality Criterion\n\n')
        file.write('GROUP\n')
        file.write('1 1 P 1\n\n')
        file.write('CELLP\n')
        file.write(f'{structure.lattice.a} {structure.lattice.b} {structure.lattice.c} {structure.lattice.alpha} {structure.lattice.beta} {structure.lattice.gamma}\n')
        file.write('  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n')
        file.write('STRUC\n')
        for isite, site in enumerate(structure, start=1):
            file.write(f'{isite} {site.species_string} {site.species_string}{isite} 1.0000 {site.frac_coords[0]} {site.frac_coords[1]} {site.frac_coords[2]} 1a 1\n')
            file.write(' 0.000000 0.000000 0.000000 0.00\n')
        file.write('  0 0 0 0 0 0 0\n')
        file.write('THERT 0\n')
        file.write('THERM\n')
        counter = 1
        for atom_therm, site in zip(matrix_cif, structure):
            file.write(f'{counter} {site.species_string}{counter} {atom_therm[0]} {atom_therm[1]} {atom_therm[2]} {atom_therm[5]} {atom_therm[4]} {atom_therm[3]}\n')
            counter += 1
        file.write('  0 0 0 0 0 0 0 0\n')
        file.write('VECTR\n')
        vector_count = 1
        site_count = 1
        for vectors in result:
            vector0_x = vectors['vector0'][0]
            vector0_y = vectors['vector0'][1]
            vector0_z = vectors['vector0'][2]
            vector1_x = vectors['vector1'][0]
            vector1_y = vectors['vector1'][1]
            vector1_z = vectors['vector1'][2]
            file.write(f'    {vector_count} {vector0_x} {vector0_y} {vector0_z} 0\n')
            file.write(f'    {site_count} 0 0 0 0\n')
            file.write(' 0 0 0 0 0\n')
            vector_count += 1
            file.write(f'    {vector_count} {vector1_x} {vector1_y} {vector1_z} 0\n')
            file.write(f'    {site_count} 0 0 0 0\n')
            vector_count += 1
            site_count += 1
            file.write(' 0 0 0 0 0\n')
        file.write(' 0 0 0 0 0\n')
        file.write('VECTT\n')
        counter = 1
        for _i in range(len(result)):
            file.write(f'{counter} 0.2 255 0 0 1\n')
            counter += 1
            file.write(f'{counter} 0.2 0 0 255 1\n')
            counter += 1
        file.write(' 0 0 0 0 0\n')