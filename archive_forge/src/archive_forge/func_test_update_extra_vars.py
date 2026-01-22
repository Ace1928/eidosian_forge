import tempfile
from pecan.templating import RendererFactory, format_line_context
from pecan.tests import PecanTestCase
def test_update_extra_vars(self):
    extra_vars = self.rf.extra_vars
    extra_vars.update({'foo': 1})
    self.assertEqual(extra_vars.make_ns({'bar': 2}), {'foo': 1, 'bar': 2})
    self.assertEqual(extra_vars.make_ns({'foo': 2}), {'foo': 2})