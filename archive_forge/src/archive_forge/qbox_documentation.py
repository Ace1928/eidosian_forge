from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader
import re
import xml.etree.ElementTree as ET
Parse a certain frame from QBOX output

    Inputs:
        tree - ElementTree, <iteration> block from output file
        species - dict, data about species. Key is name of atom type,
            value is data about that type
    Return:
        Atoms object describing this iteration