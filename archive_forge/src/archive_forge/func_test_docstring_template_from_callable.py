import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def test_docstring_template_from_callable():
    docstr = textwrap.dedent('\n        This is the func_d method.\n\n        It computes D.\n        ')
    assert func_d.__doc__ == docstr