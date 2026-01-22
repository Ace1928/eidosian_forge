from sympy.core.numbers import (I, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import (And, Or)
from sympy.plotting.plot_implicit import plot_implicit
from sympy.plotting.plot import unset_show
from tempfile import NamedTemporaryFile, mkdtemp
from sympy.testing.pytest import skip, warns, XFAIL
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
import os
def test_region_and():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if not matplotlib:
        skip('Matplotlib not the default backend')
    from matplotlib.testing.compare import compare_images
    test_directory = os.path.dirname(os.path.abspath(__file__))
    try:
        temp_dir = mkdtemp()
        TmpFileManager.tmp_folder(temp_dir)
        x, y = symbols('x y')
        r1 = (x - 1) ** 2 + y ** 2 < 2
        r2 = (x + 1) ** 2 + y ** 2 < 2
        test_filename = tmp_file(dir=temp_dir, name='test_region_and')
        cmp_filename = os.path.join(test_directory, 'test_region_and.png')
        p = plot_implicit(r1 & r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
        test_filename = tmp_file(dir=temp_dir, name='test_region_or')
        cmp_filename = os.path.join(test_directory, 'test_region_or.png')
        p = plot_implicit(r1 | r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
        test_filename = tmp_file(dir=temp_dir, name='test_region_not')
        cmp_filename = os.path.join(test_directory, 'test_region_not.png')
        p = plot_implicit(~r1, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
        test_filename = tmp_file(dir=temp_dir, name='test_region_xor')
        cmp_filename = os.path.join(test_directory, 'test_region_xor.png')
        p = plot_implicit(r1 ^ r2, x, y)
        p.save(test_filename)
        compare_images(cmp_filename, test_filename, 0.005)
    finally:
        TmpFileManager.cleanup()