import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def perpendicular_vector(vector, base):
    """Remove the components of *vector* that are parallel to *base*"""
    return vector - parallel_vector(vector, base)