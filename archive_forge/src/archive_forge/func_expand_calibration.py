from copy import copy
from dataclasses import dataclass
from typing import Union, Dict, List, Any, Optional, no_type_check
from pyquil.quilatom import (
from pyquil.quilbase import (
def expand_calibration(match: CalibrationMatch) -> List[AbstractInstruction]:
    """ " Expand the body of a calibration from a match."""
    return [fill_placeholders(instr, match.settings) for instr in match.cal.instrs]