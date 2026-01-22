import importlib
import logging
import re
from io import StringIO
from xml.dom import minidom
from xml.parsers.expat import ExpatError
from rdkit.Chem import Mol
@staticmethod
def is_mol(x):
    """Return True if x is a Chem.Mol"""
    return isinstance(x, Mol)