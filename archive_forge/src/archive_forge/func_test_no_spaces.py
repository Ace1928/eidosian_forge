from ... import tests
from .. import rio
def test_no_spaces(self):
    self.assertFalse(self.module._valid_tag('foo bla'))