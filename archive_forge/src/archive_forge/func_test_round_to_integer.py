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
@pytest.mark.parametrize('ty', ['round', 'round_to_multiple'])
def test_round_to_integer(ty):
    if ty == 'round':
        round = pc.round
        RoundOptions = partial(pc.RoundOptions, ndigits=0)
    elif ty == 'round_to_multiple':
        round = pc.round_to_multiple
        RoundOptions = partial(pc.RoundToMultipleOptions, multiple=1)
    values = [3.2, 3.5, 3.7, 4.5, -3.2, -3.5, -3.7, None]
    rmode_and_expected = {'down': [3, 3, 3, 4, -4, -4, -4, None], 'up': [4, 4, 4, 5, -3, -3, -3, None], 'towards_zero': [3, 3, 3, 4, -3, -3, -3, None], 'towards_infinity': [4, 4, 4, 5, -4, -4, -4, None], 'half_down': [3, 3, 4, 4, -3, -4, -4, None], 'half_up': [3, 4, 4, 5, -3, -3, -4, None], 'half_towards_zero': [3, 3, 4, 4, -3, -3, -4, None], 'half_towards_infinity': [3, 4, 4, 5, -3, -4, -4, None], 'half_to_even': [3, 4, 4, 4, -3, -4, -4, None], 'half_to_odd': [3, 3, 4, 5, -3, -3, -4, None]}
    for round_mode, expected in rmode_and_expected.items():
        options = RoundOptions(round_mode=round_mode)
        result = round(values, options=options)
        np.testing.assert_array_equal(result, pa.array(expected))