from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def get_gluings(tet):
    """
            Given a tetrahedron, return the four face gluings.
            """
    return [tet.getAdjacentTetrahedronGluing(face) for face in range(4)]