import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_function_exception_key_error(self):
    func1 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
    expected = '%s %s' % ('<heat.tests.test_function.TestFunctionKeyError', "{foo: ['bar', 'baz']} -> ???>")
    self.assertEqual(expected, str(func1))