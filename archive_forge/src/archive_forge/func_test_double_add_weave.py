from .. import errors as errors
from .. import identitymap as identitymap
from . import TestCase
def test_double_add_weave(self):
    map = identitymap.NullIdentityMap()
    weave = 'foo'
    map.add_weave('id', weave)
    map.add_weave('id', weave)
    self.assertEqual(None, map.find_weave('id'))