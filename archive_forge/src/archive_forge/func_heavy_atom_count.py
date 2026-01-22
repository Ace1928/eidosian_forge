import os
import re
import sqlite3
import subprocess
import sys
from optparse import OptionParser
from rdkit import Chem
def heavy_atom_count(smi):
    m = Chem.MolFromSmiles(smi)
    return m.GetNumAtoms()