import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def test_escaping_in_docstring():
    docstr = textwrap.dedent('\n        This is the func_f method.\n\n        { We can escape curly braces like this. }\n\n        Examples\n        --------\n        We should replace curly brace usage in doctests.\n\n        >>> dict(x = "x", y = "y")\n        >>> set((1, 2, 3))\n        ')
    assert func_f.__doc__ == docstr