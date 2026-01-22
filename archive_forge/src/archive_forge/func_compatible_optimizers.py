import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
@staticmethod
def compatible_optimizers():
    """
        List of compatible optimizers.
        """
    return ['adam', 'mem_eff_adam', 'adafactor']