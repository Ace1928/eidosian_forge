import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def enabled_output_names(self):
    """Return the set of names of all enabled outputs in the template."""
    if self._output_defns is None:
        self._load_output_defns()
    return set(self._output_defns)