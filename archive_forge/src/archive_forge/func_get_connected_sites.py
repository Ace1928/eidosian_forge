from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
def get_connected_sites(self, n):
    """
        Returns a named tuple of neighbors of site n:
        periodic_site, jimage, index, weight.
        Index is the index of the corresponding site
        in the original structure, weight can be
        None if not defined.
        Args:
            n: index of Site in Molecule
            jimage: lattice vector of site

        Returns:
            list of ConnectedSite tuples,
            sorted by closest first.
        """
    connected_sites = set()
    out_edges = list(self.graph.out_edges(n, data=True))
    in_edges = list(self.graph.in_edges(n, data=True))
    for u, v, d in out_edges + in_edges:
        weight = d.get('weight')
        if v == n:
            site = self.molecule[u]
            dist = self.molecule[v].distance(self.molecule[u])
            connected_site = ConnectedSite(site=site, jimage=(0, 0, 0), index=u, weight=weight, dist=dist)
        else:
            site = self.molecule[v]
            dist = self.molecule[u].distance(self.molecule[v])
            connected_site = ConnectedSite(site=site, jimage=(0, 0, 0), index=v, weight=weight, dist=dist)
        connected_sites.add(connected_site)
    connected_sites = list(connected_sites)
    connected_sites.sort(key=lambda x: x.dist)
    return connected_sites