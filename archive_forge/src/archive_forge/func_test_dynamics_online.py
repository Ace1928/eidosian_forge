import os
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev
def test_dynamics_online():
    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples', 'dynamics-online')
    if os.path.isdir(dir_path):
        ch1 = ['1-4', '1-5', '1-6', '1-7', '1-8', '1-9_1', '1-9_2', '1-9_3']
        ch2 = ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', 'circular']
        ch3 = ['3-1_1', '3-1_2', '3-2_1', '3-2_2', '3-2_3', '3-2_4', '3-2_5', '3-3']
        ch4 = ['4-1_1', '4-2_1', '4-4_1', '4-4_2', '4-5_1', '4-5_2']
        chapters = [(ch1, 'ch1'), (ch2, 'ch2'), (ch3, 'ch3'), (ch4, 'ch4')]
        for ch, name in chapters:
            for i in ch:
                in_filepath = os.path.join('dynamics-online', name, i + '.al')
                out_filepath = os.path.join('dynamics-online', name, i + '.py')
                _test_examples(in_filepath, out_filepath, i)