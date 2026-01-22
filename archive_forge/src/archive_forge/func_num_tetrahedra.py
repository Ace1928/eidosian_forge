from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities
def num_tetrahedra(self):
    """
        Returns the number of tetrahedra
        """
    return self.getNumberOfTetrahedra()