from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_round_to_multiple():
    values = [320, 3.5, 3.075, 4.5, -3.212, -35.1234, -3.045, None]
    multiple_and_expected = {0.05: [320, 3.5, 3.1, 4.5, -3.2, -35.1, -3.05, None], pa.scalar(0.1): [320, 3.5, 3.1, 4.5, -3.2, -35.1, -3, None], 2: [320, 4, 4, 4, -4, -36, -4, None], 10: [320, 0, 0, 0, -0, -40, -0, None], pa.scalar(100, type=pa.decimal256(10, 4)): [300, 0, 0, 0, -0, -0, -0, None]}
    for multiple, expected in multiple_and_expected.items():
        options = pc.RoundToMultipleOptions(multiple, 'half_towards_infinity')
        result = pc.round_to_multiple(values, options=options)
        np.testing.assert_allclose(result, pa.array(expected), equal_nan=True)
        assert pc.round_to_multiple(values, multiple, 'half_towards_infinity') == result
    for multiple in [0, -2, pa.scalar(-10.4)]:
        with pytest.raises(pa.ArrowInvalid, match='Rounding multiple must be positive'):
            pc.round_to_multiple(values, multiple=multiple)
    for multiple in [object, 99999999999999999999999]:
        with pytest.raises(TypeError, match='is not a valid multiple type'):
            pc.round_to_multiple(values, multiple=multiple)