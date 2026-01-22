import types
from _pydev_bundle import pydev_log
from typing import Tuple, Literal
def required_events_stepping(self) -> Tuple[Literal['line', 'call', 'return'], ...]:
    ret = ()
    for plugin in self.active_plugins:
        new = plugin.required_events_stepping()
        if new:
            ret += new
    return ret