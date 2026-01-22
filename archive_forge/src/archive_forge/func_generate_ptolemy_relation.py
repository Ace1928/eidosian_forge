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
def generate_ptolemy_relation(tet, index):

    def generate_Ptolemy_coordinate(addl_index):
        total_index = matrix.vector_add(index, addl_index)
        return Polynomial.from_variable_name('c_%d%d%d%d' % tuple(total_index) + '_%d' % tet)

    def generate_obstruction_variable(face):
        if has_obstruction_class:
            return Polynomial.from_variable_name('s_%d_%d' % (face, tet))
        else:
            return Polynomial.constant_polynomial(1)
    return generate_obstruction_variable(0) * generate_obstruction_variable(1) * generate_Ptolemy_coordinate((1, 1, 0, 0)) * generate_Ptolemy_coordinate((0, 0, 1, 1)) - generate_obstruction_variable(0) * generate_obstruction_variable(2) * generate_Ptolemy_coordinate((1, 0, 1, 0)) * generate_Ptolemy_coordinate((0, 1, 0, 1)) + generate_obstruction_variable(0) * generate_obstruction_variable(3) * generate_Ptolemy_coordinate((1, 0, 0, 1)) * generate_Ptolemy_coordinate((0, 1, 1, 0))