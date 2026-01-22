from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test__name_levels():
    assert _name_levels('a', ['b', 'c']) == ['[ab]', '[ac]']