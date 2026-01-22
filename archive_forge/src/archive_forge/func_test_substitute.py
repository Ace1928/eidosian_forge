import pytest
import sys
from .roundtrip import round_trip, dedent, round_trip_load, round_trip_dump
def test_substitute(self):
    x = '\n        args:\n          username: anthon          # name\n          passwd: secret            # password\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test\n          loop:\n            wait: 10\n        '
    data = round_trip_load(x)
    data['args']['passwd'] = 'deleted password'
    x = x.replace(': secret          ', ': deleted password')
    assert round_trip_dump(data) == dedent(x)