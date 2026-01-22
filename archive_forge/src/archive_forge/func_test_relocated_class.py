import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_relocated_class(self):
    warning = "DEPRECATED: the 'myFoo' class has been moved to 'pyomo.common.tests.relocated.Bar'"
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo'):
        from pyomo.common.tests.test_deprecated import myFoo
    self.assertEqual(myFoo.data, 42)
    self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
    from pyomo.common.tests import relocated
    self.assertNotIn('Foo', dir(relocated))
    self.assertNotIn('Foo_2', dir(relocated))
    warning = "DEPRECATED: the 'Foo_2' class has been moved to 'pyomo.common.tests.relocated.Bar'"
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo'):
        self.assertIs(relocated.Foo_2, relocated.Bar)
        self.assertEqual(relocated.Foo_2.data, 42)
    self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
    self.assertNotIn('Foo', dir(relocated))
    self.assertIn('Foo_2', dir(relocated))
    self.assertIs(relocated.Foo_2, relocated.Bar)
    warning = "DEPRECATED: the 'Foo' class has been moved to 'pyomo.common.tests.test_deprecated.Bar'"
    OUT = StringIO()
    with LoggingIntercept(OUT, 'pyomo'):
        from pyomo.common.tests.relocated import Foo
        self.assertEqual(Foo.data, 21)
    self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
    self.assertIn('Foo', dir(relocated))
    self.assertIn('Foo_2', dir(relocated))
    self.assertIs(relocated.Foo, Bar)
    with self.assertRaisesRegex(AttributeError, "(?:module 'pyomo.common.tests.relocated')|(?:'module' object) has no attribute 'Baz'"):
        relocated.Baz.data
    self.assertEqual(relocated.Foo_3, '_3')
    with self.assertRaisesRegex(AttributeError, "(?:module 'pyomo.common.tests.test_deprecated')|(?:'module' object) has no attribute 'Baz'"):
        sys.modules[__name__].Baz.data