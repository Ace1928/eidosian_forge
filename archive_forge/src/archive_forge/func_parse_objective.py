import operator
import os
import sys
import warnings
from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError
from .core import scip_path, fscip_path
from .. import constants
from typing import Dict, List, Optional, Tuple
@staticmethod
def parse_objective(string: str) -> Optional[float]:
    fields = string.split(':')
    if len(fields) != 2:
        return None
    label, objective = fields
    if label != 'objective value':
        return None
    objective = objective.strip()
    try:
        objective = float(objective)
    except ValueError:
        return None
    return objective