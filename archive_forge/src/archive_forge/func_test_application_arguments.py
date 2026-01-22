import pytest  # NOQA
from .roundtrip import round_trip
def test_application_arguments(self):
    round_trip('\n        args:\n          username: anthon\n          passwd: secret\n          fullname: Anthon van der Neut\n          tmux:\n            session-name: test\n          loop:\n            wait: 10\n        ')