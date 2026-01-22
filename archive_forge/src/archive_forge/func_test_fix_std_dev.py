from builtins import str
import sys
import os
def test_fix_std_dev():
    """Tests the transformation of std_dev() into std_dev."""
    tests = {'x.std_dev()': 'x.std_dev', 'y.std_dev();  unc.std_dev(z)': 'y.std_dev;  unc.std_dev(z)', 'uncertainties.std_dev(x)': 'uncertainties.std_dev(x)', 'std_dev(x)': 'std_dev(x)', 'obj.x.std_dev()': 'obj.x.std_dev', '\n            long_name.std_dev(\n            # No argument!\n            )': '\n            long_name.std_dev', 'x.set_std_dev(3)': 'x.std_dev = 3', 'y = set_std_dev(3)': 'y = set_std_dev(3)', 'func = x.set_std_dev': 'func = x.set_std_dev', 'obj.x.set_std_dev(sin(y))': 'obj.x.std_dev = sin(y)'}
    check_all('std_dev', tests)