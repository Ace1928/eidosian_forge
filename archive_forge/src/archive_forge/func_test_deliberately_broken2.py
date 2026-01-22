import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
@dec.skip('Testing the skip decorator')
def test_deliberately_broken2():
    """Another deliberately broken test - we want to skip this one."""
    1 / 0