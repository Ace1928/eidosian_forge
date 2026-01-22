import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('sparse_columns, exp_cols', [(True, [{'is_visible': True, 'attributes': 'colspan="2"', 'value': 'c0'}, {'is_visible': False, 'attributes': '', 'value': 'c0'}]), (False, [{'is_visible': True, 'attributes': '', 'value': 'c0'}, {'is_visible': True, 'attributes': '', 'value': 'c0'}])])
def test_mi_styler_sparsify_columns(mi_styler, sparse_columns, exp_cols):
    exp_l1_c0 = {'is_visible': True, 'attributes': '', 'display_value': 'c1_a'}
    exp_l1_c1 = {'is_visible': True, 'attributes': '', 'display_value': 'c1_b'}
    ctx = mi_styler._translate(True, sparse_columns)
    assert exp_cols[0].items() <= ctx['head'][0][2].items()
    assert exp_cols[1].items() <= ctx['head'][0][3].items()
    assert exp_l1_c0.items() <= ctx['head'][1][2].items()
    assert exp_l1_c1.items() <= ctx['head'][1][3].items()