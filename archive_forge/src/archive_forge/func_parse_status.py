import operator
import os
import sys
import warnings
from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError
from .core import scip_path, fscip_path
from .. import constants
from typing import Dict, List, Optional, Tuple
@staticmethod
def parse_status(string: str) -> Optional[int]:
    for fscip_status, pulp_status in FSCIP_CMD.FSCIP_STATUSES.items():
        if fscip_status in string:
            return pulp_status
    return None