import collections
import math
from typing import (
from pip._vendor.resolvelib.providers import AbstractProvider
from .base import Candidate, Constraint, Requirement
from .candidates import REQUIRES_PYTHON_IDENTIFIER
from .factory import Factory
@staticmethod
def is_backtrack_cause(identifier: str, backtrack_causes: Sequence['PreferenceInformation']) -> bool:
    for backtrack_cause in backtrack_causes:
        if identifier == backtrack_cause.requirement.name:
            return True
        if backtrack_cause.parent and identifier == backtrack_cause.parent.name:
            return True
    return False