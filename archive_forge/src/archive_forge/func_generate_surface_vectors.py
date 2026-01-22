from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from pymatgen.analysis.elasticity.strain import Deformation, Strain
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, ZSLMatch, reduce_vectors
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
def generate_surface_vectors(self, film: Structure, substrate: Structure, film_millers: ArrayLike, substrate_millers: ArrayLike):
    """
        Generates the film/substrate slab combinations for a set of given
        miller indices.

        Args:
            film (Structure): film structure
            substrate (Structure): substrate structure
            film_millers (array): all miller indices to generate slabs for
                film
            substrate_millers (array): all miller indices to generate slabs
                for substrate
        """
    vector_sets = []
    for f_miller in film_millers:
        film_slab = SlabGenerator(film, f_miller, 20, 15, primitive=False).get_slab()
        film_vectors = reduce_vectors(film_slab.oriented_unit_cell.lattice.matrix[0], film_slab.oriented_unit_cell.lattice.matrix[1])
        for s_miller in substrate_millers:
            substrate_slab = SlabGenerator(substrate, s_miller, 20, 15, primitive=False).get_slab()
            substrate_vectors = reduce_vectors(substrate_slab.oriented_unit_cell.lattice.matrix[0], substrate_slab.oriented_unit_cell.lattice.matrix[1])
            vector_sets.append((film_vectors, substrate_vectors, f_miller, s_miller))
    return vector_sets