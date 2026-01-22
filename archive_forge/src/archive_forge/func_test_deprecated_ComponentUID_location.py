import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_deprecated_ComponentUID_location(self):
    import pyomo.core.base.component as comp
    self.assertNotIn('ComponentUID', dir(comp))
    warning = "DEPRECATED: the 'ComponentUID' class has been moved to 'pyomo.core.base.componentuid.ComponentUID'"
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo.core'):
        from pyomo.core.base.component import ComponentUID as old_ComponentUID
    self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
    self.assertIs(old_ComponentUID, ComponentUID)
    self.assertIs(old_ComponentUID, ComponentUID)
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo.core'):
        self.assertIs(comp.ComponentUID, ComponentUID)
    self.assertEqual('', OUT.getvalue())