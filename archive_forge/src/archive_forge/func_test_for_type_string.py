from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
def test_for_type_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    assert f.for_type(type_str, foo_printer) is None
    assert f.for_type(type_str) is foo_printer
    assert _mod_name_key(C) in f.deferred_printers
    assert f.for_type(C) is foo_printer
    assert _mod_name_key(C) not in f.deferred_printers
    assert C in f.type_printers