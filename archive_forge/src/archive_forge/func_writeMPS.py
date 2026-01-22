from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def writeMPS(self, filename, mpsSense=0, rename=0, mip=1, with_objsense: bool=False):
    """
        Writes an mps files from the problem information

        :param str filename: name of the file to write
        :param int mpsSense:
        :param bool rename: if True, normalized names are used for variables and constraints
        :param mip: variables and variable renames
        :return:
        Side Effects:
            - The file is created
        """
    return mpslp.writeMPS(self, filename, mpsSense=mpsSense, rename=rename, mip=mip, with_objsense=with_objsense)