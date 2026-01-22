from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_nested_badkey(self):
    d = {'x': 1, 'y': 2, 'z': (sum, ['x', 'y'])}
    try:
        result = self.get(d, [['badkey'], 'y'])
    except KeyError:
        pass
    else:
        msg = 'Expected `{}` with badkey to raise KeyError.\n'
        msg += f"Obtained '{result}' instead."
        assert False, msg.format(self.get.__name__)