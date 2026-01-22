import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def read_ase_and_abinit_inputs(label):
    with open(label + '.in') as fd:
        atoms = read_abinit_in(fd)
    parameters = Parameters.read(label + '.ase')
    return (atoms, parameters)