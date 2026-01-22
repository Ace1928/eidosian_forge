import linecache
import sys
from IPython.core import compilerop
def test_code_name():
    code = 'x=1'
    name = compilerop.code_name(code)
    assert name.startswith('<ipython-input-0')