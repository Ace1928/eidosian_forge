from heat.common import exception
from heat.engine import dependencies
from heat.tests import common
def test_single_fwd(self):
    self._dep_test_fwd(('second', 'first'))