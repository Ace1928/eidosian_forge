from __future__ import print_function
import pytest
import platform
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump  # NOQA
@pytest.mark.skipif(platform.python_implementation() == 'Jython', reason='Jython throws RepresenterError')
def test_preserve_string_strip(self):
    s = '\n        a: |-\n          abc\n          def\n\n        '
    round_trip(s, intermediate=dict(a='abc\ndef'))