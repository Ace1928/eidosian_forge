import os
import subprocess
import tempfile
import warnings
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Selection
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import is_aa
from Bio import BiopythonWarning
def ca_depth(residue, surface):
    """Return CA depth."""
    if not residue.has_id('CA'):
        return None
    ca = residue['CA']
    coord = ca.get_coord()
    return min_dist(coord, surface)