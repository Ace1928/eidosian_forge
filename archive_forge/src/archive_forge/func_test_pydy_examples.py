import os
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev
def test_pydy_examples():
    l = ['mass_spring_damper', 'chaos_pendulum', 'double_pendulum', 'non_min_pendulum']
    for i in l:
        in_filepath = os.path.join('pydy-example-repo', i + '.al')
        out_filepath = os.path.join('pydy-example-repo', i + '.py')
        _test_examples(in_filepath, out_filepath, i)