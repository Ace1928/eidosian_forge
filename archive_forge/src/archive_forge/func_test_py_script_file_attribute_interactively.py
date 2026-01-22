import unittest
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
@dec.skip_win32
def test_py_script_file_attribute_interactively(self):
    """Test that `__file__` is not set after `ipython -i file.py`"""
    src = 'True\n'
    self.mktmp(src)
    out, err = tt.ipexec(self.fname, options=['-i'], commands=['"__file__" in globals()', 'print(123)', 'exit()'])
    assert 'False' in out, f'Subprocess stderr:\n{err}\n-----'