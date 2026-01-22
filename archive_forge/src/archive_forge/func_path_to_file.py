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
def path_to_file(self):
    name = self._manifold.name()
    if re.match('([msvt]|o9_)[0-9]+$', name):
        dir = 'OrientableCuspedCensus'
    elif re.match('[0-9]+([\\^][0-9]+)?[_][0-9]+$', name):
        dir = 'LinkExteriors'
    elif re.match('[KL][0-9]+[an][0-9]+$', name):
        dir = 'HTLinkExteriors'
    else:
        raise Exception('No canonical path for manifold')
    tets = self._manifold.num_tetrahedra()
    return '/'.join(['data', 'pgl%d' % self._N, dir, '%02d_tetrahedra' % tets])