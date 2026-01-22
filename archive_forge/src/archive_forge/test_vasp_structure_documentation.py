import os
import numpy as np
import numpy.testing
import unittest
import ase
import ase.build
import ase.io
from ase.io.vasp import write_vasp_xdatcar
from ase.calculators.calculator import compare_atoms
Compare two Atoms objects, raising AssertionError if different