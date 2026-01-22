import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_no_end_of_file_eol(self):
    """not excluding comments caused some problems if at the end of
        the file without a newline. First error, then included \x00 """
    x = '        - europe: 10 # abc'
    round_trip(x, extra='\n')
    with pytest.raises(AssertionError):
        round_trip(x, extra='a\n')