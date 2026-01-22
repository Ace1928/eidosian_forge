from typing import (
import cmath
import re
import numpy as np
import sympy
def is_valid_end_token(tok: _HangingToken) -> bool:
    return tok != '(' and (not isinstance(tok, _CustomQuirkOperationToken))