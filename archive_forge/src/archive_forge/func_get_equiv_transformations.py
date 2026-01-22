from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.util.due import Doi, due
from pymatgen.util.numba import njit
def get_equiv_transformations(self, transformation_sets, film_vectors, substrate_vectors):
    """
        Applies the transformation_sets to the film and substrate vectors
        to generate super-lattices and checks if they matches.
        Returns all matching vectors sets.

        Args:
            transformation_sets(array): an array of transformation sets:
                each transformation set is an array with the (i,j)
                indicating the area multiples of the film and substrate it
                corresponds to, an array with all possible transformations
                for the film area multiple i and another array for the
                substrate area multiple j.
            film_vectors(array): film vectors to generate super lattices
            substrate_vectors(array): substrate vectors to generate super
                lattices
        """
    for film_transformations, substrate_transformations in transformation_sets:
        films = np.array([reduce_vectors(*v) for v in np.dot(film_transformations, film_vectors)], dtype=float)
        substrates = np.array([reduce_vectors(*v) for v in np.dot(substrate_transformations, substrate_vectors)], dtype=float)
        for (f_trans, s_trans), (f, s) in zip(product(film_transformations, substrate_transformations), product(films, substrates)):
            if is_same_vectors(f, s, bidirectional=self.bidirectional, max_length_tol=self.max_length_tol, max_angle_tol=self.max_angle_tol):
                yield [f, s, f_trans, s_trans]