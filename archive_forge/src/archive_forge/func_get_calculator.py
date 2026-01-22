import argparse
import traceback
from math import pi
from time import time
import numpy as np
import ase.db
import ase.optimize
from ase.calculators.emt import EMT
from ase.io import Trajectory
def get_calculator(self):
    return self.atoms.calc