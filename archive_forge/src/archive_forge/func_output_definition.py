import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def output_definition(self, output_name):
    """Return the definition of the given output."""
    if self._output_defns is None:
        self._load_output_defns()
    return self._output_defns[output_name]