import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def get_vasp_version(string):
    """Extract version number from header of stdout.

    Example::

      >>> get_vasp_version('potato vasp.6.1.2 bumblebee')
      '6.1.2'

    """
    match = re.search('vasp\\.(\\S+)', string, re.M)
    return match.group(1)