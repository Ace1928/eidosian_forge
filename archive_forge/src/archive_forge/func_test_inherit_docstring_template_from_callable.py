import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def test_inherit_docstring_template_from_callable():
    docstr = textwrap.dedent('\n        This is the func_e method.\n\n        It computes E.\n        ')
    assert func_e.__doc__ == docstr