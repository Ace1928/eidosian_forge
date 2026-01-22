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

        In general, there can be d different solutions to the (reduced) Ptolemy
        variety for each solution to the gluing equations (with peripheral
        equations). This method computes d which is also the number of elements
        in H^1(Mhat; Z/N) where Mhat is the non-manifold cell comples obtained
        by gluing together the tetrahedra as non-ideal tetrahedra.


        For example, the Ptolemy variety for m009 has 4 points but there are
        only 2 distinct boundary-unipotent PSL(2,C) representations.
        Thus the following call returns 2 = 4 / 2

        >>> from snappy import Manifold
        >>> Manifold("m009").ptolemy_variety(2,1).degree_to_shapes()
        2

        >>> Manifold("m010").ptolemy_variety(2,1).degree_to_shapes()
        2
        >>> Manifold("m011").ptolemy_variety(2,1).degree_to_shapes()
        1

        >>> Manifold("m009").ptolemy_variety(3,1).degree_to_shapes()
        1
        >>> Manifold("m010").ptolemy_variety(3,1).degree_to_shapes()
        3
        >>> Manifold("m011").ptolemy_variety(3,1).degree_to_shapes()
        1

        