from cliff import columns
from osc_lib import utils
def machine_readable(self):
    return [dict(x) for x in self._value or []]