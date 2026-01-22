from openstack import _hacking
from openstack.tests.unit import base
def test_assert_no_setupclass(self):
    self.assertEqual(len(list(_hacking.assert_no_setupclass('def setUpClass(cls)'))), 1)
    self.assertEqual(len(list(_hacking.assert_no_setupclass('# setUpClass is evil'))), 0)
    self.assertEqual(len(list(_hacking.assert_no_setupclass('def setUpClassyDrinkingLocation(cls)'))), 0)