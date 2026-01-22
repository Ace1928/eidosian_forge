from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient.v3 import volume_types
def test_update_name(self):
    """Test volume_type update shell command

        Verify that only name is updated and the description and
        is_public properties remains unchanged.
        """
    t = cs.volume_types.create('test-type-3', 'test_type-3-desc', True)
    self.assertTrue(t.is_public)
    t1 = cs.volume_types.update(t.id, 'test-type-2')
    cs.assert_called('PUT', '/types/3', {'volume_type': {'name': 'test-type-2', 'description': None}})
    self.assertEqual('test-type-2', t1.name)
    self.assertEqual('test_type-3-desc', t1.description)
    self.assertTrue(t1.is_public)