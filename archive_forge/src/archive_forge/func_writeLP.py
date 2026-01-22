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
def writeLP(self, filename, writeSOS=1, mip=1, max_length=100):
    """
        Write the given Lp problem to a .lp file.

        This function writes the specifications (objective function,
        constraints, variables) of the defined Lp problem to a file.

        :param str filename: the name of the file to be created.
        :return: variables
        Side Effects:
            - The file is created
        """
    return mpslp.writeLP(self, filename=filename, writeSOS=writeSOS, mip=mip, max_length=max_length)