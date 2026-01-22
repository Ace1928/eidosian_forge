from __future__ import annotations
import itertools
import logging
import time
from random import shuffle
from typing import TYPE_CHECKING
import numpy as np
from numpy.linalg import norm, svd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.core import Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.due import Doi, due
def setup_test_perfect_environment(self, symbol, randomness=False, max_random_dist=0.1, symbol_type='mp_symbol', indices='RANDOM', random_translation='NONE', random_rotation='NONE', random_scale='NONE', points=None):
    """
        Args:
            symbol:
            randomness:
            max_random_dist:
            symbol_type:
            indices:
            random_translation:
            random_rotation:
            random_scale:
            points:
        """
    if symbol_type == 'IUPAC':
        cg = self.allcg.get_geometry_from_IUPAC_symbol(symbol)
    elif symbol_type in ('MP', 'mp_symbol'):
        cg = self.allcg.get_geometry_from_mp_symbol(symbol)
    elif symbol_type == 'CoordinationGeometry':
        cg = symbol
    else:
        raise ValueError('Wrong mp_symbol to setup coordination geometry')
    neighb_coords = []
    _points = points if points is not None else cg.points
    if randomness:
        rv = np.random.random_sample(3)
        while norm(rv) > 1.0:
            rv = np.random.random_sample(3)
        coords = [np.zeros(3, float) + max_random_dist * rv]
        for pp in _points:
            rv = np.random.random_sample(3)
            while norm(rv) > 1.0:
                rv = np.random.random_sample(3)
            neighb_coords.append(np.array(pp) + max_random_dist * rv)
    else:
        coords = [np.zeros(3, float)]
        for pp in _points:
            neighb_coords.append(np.array(pp))
    if indices == 'RANDOM':
        shuffle(neighb_coords)
    elif indices == 'ORDERED':
        pass
    else:
        neighb_coords = [neighb_coords[ii] for ii in indices]
    if random_scale == 'RANDOM':
        scale = 0.1 * np.random.random_sample() + 0.95
    elif random_scale == 'NONE':
        scale = 1.0
    else:
        scale = random_scale
    coords = [scale * cc for cc in coords]
    neighb_coords = [scale * cc for cc in neighb_coords]
    if random_rotation == 'RANDOM':
        uu = np.random.random_sample(3) + 0.1
        uu = uu / norm(uu)
        theta = np.pi * np.random.random_sample()
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        ux = uu[0]
        uy = uu[1]
        uz = uu[2]
        rand_rot = [[ux * ux + (1.0 - ux * ux) * cos_theta, ux * uy * (1.0 - cos_theta) - uz * sin_theta, ux * uz * (1.0 - cos_theta) + uy * sin_theta], [ux * uy * (1.0 - cos_theta) + uz * sin_theta, uy * uy + (1.0 - uy * uy) * cos_theta, uy * uz * (1.0 - cos_theta) - ux * sin_theta], [ux * uz * (1.0 - cos_theta) - uy * sin_theta, uy * uz * (1.0 - cos_theta) + ux * sin_theta, uz * uz + (1.0 - uz * uz) * cos_theta]]
    elif random_rotation == 'NONE':
        rand_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    else:
        rand_rot = random_rotation
    new_coords = []
    for coord in coords:
        new_cc = np.dot(rand_rot, coord).T
        new_coords.append(new_cc.ravel())
    coords = new_coords
    new_coords = []
    for coord in neighb_coords:
        new_cc = np.dot(rand_rot, coord.T)
        new_coords.append(new_cc.ravel())
    neighb_coords = new_coords
    if random_translation == 'RANDOM':
        translation = 10.0 * (2.0 * np.random.random_sample(3) - 1.0)
    elif random_translation == 'NONE':
        translation = np.zeros(3, float)
    else:
        translation = random_translation
    coords = [cc + translation for cc in coords]
    neighb_coords = [cc + translation for cc in neighb_coords]
    coords.extend(neighb_coords)
    species = ['O'] * len(coords)
    species[0] = 'Cu'
    amin = np.min([cc[0] for cc in coords])
    amax = np.max([cc[0] for cc in coords])
    bmin = np.min([cc[1] for cc in coords])
    bmax = np.max([cc[1] for cc in coords])
    cmin = np.min([cc[2] for cc in coords])
    cmax = np.max([cc[2] for cc in coords])
    factor = 5.0
    aa = factor * max([amax - amin, bmax - bmin, cmax - cmin])
    lattice = Lattice.cubic(a=aa)
    structure = Structure(lattice=lattice, species=species, coords=coords, to_unit_cell=False, coords_are_cartesian=True)
    self.setup_structure(structure=structure)
    self.setup_local_geometry(isite=0, coords=neighb_coords)
    self.perfect_geometry = AbstractGeometry.from_cg(cg=cg)