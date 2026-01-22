import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def vf_to_vv(self, vertices, simplices):
    """
        Convert a vertex-face mesh to a vertex-vertex mesh used by this class

        Parameters
        ----------
        vertices : list
            Vertices
        simplices : list
            Simplices
        """
    if self.dim > 1:
        for s in simplices:
            edges = itertools.combinations(s, self.dim)
            for e in edges:
                self.V[tuple(vertices[e[0]])].connect(self.V[tuple(vertices[e[1]])])
    else:
        for e in simplices:
            self.V[tuple(vertices[e[0]])].connect(self.V[tuple(vertices[e[1]])])
    return