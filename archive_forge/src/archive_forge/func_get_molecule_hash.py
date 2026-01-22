from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
def get_molecule_hash(self, mol):
    """Return inchi as molecular hash."""
    ob_mol = BabelMolAdaptor(mol).openbabel_mol
    return self._inchi_labels(ob_mol)[2]