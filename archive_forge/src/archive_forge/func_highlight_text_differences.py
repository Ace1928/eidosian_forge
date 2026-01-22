from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from collections import defaultdict
import itertools
import random
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols, qis
from cirq.testing import lin_alg_utils
def highlight_text_differences(actual: str, expected: str) -> str:
    diff = ''
    for actual_line, desired_line in itertools.zip_longest(actual.splitlines(), expected.splitlines(), fillvalue=''):
        diff += ''.join((a if a == b else '█' for a, b in itertools.zip_longest(actual_line, desired_line, fillvalue=''))) + '\n'
    return diff