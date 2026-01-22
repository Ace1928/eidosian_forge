from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def to_magma_file(self, filename, template_path='magma/default.magma_template'):
    """
        >>> import os, tempfile
        >>> from snappy import Manifold
        >>> handle, name = tempfile.mkstemp()
        >>> p = Manifold("4_1").ptolemy_variety(2, obstruction_class=1)
        >>> p.to_magma_file(name)
        >>> os.close(handle); os.remove(name)
        """
    with open(filename, 'wb') as output:
        output.write(bytes(self.to_magma(template_path=template_path).encode('ascii')))