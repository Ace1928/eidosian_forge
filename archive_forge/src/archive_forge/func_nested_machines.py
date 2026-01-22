import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
@property
def nested_machines(self):
    """Dictionary of **all** nested state machines this machine may use."""
    return self._nested_machines