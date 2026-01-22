from oslo_config import cfg
from oslo_db import options
from oslo_utils.fixture import uuidsentinel as uuids
from glance.common import exception
from glance import context as glance_context
import glance.db.sqlalchemy.api
from glance.db.sqlalchemy import models as db_models
from glance.db.sqlalchemy import models_metadef as metadef_models
import glance.tests.functional.db as db_tests
from glance.tests.functional.db import base
from glance.tests.functional.db import base_metadef
def test_update_drop_update(self):
    """Try to create, delete, re-create property atomically.

        If we fail to undelete and claim the property, this will
        fail as duplicate.
        """
    self.db_api.image_set_property_atomic(self.image['id'], 'test_property', 'foo')
    image = self.db_api.image_get(self.adm_context, self.image['id'])
    self.assertEqual({'speed': '88mph', 'test_property': 'foo'}, self._propdict(image['properties']))
    self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'foo')
    new_props = self._propdict(image['properties'])
    del new_props['test_property']
    self.db_api.image_update(self.adm_context, self.image['id'], values={'properties': new_props}, purge_props=True)
    image = self.db_api.image_get(self.adm_context, self.image['id'])
    self.assertEqual({'speed': '88mph'}, self._propdict(image['properties']))
    self.db_api.image_set_property_atomic(self.image['id'], 'test_property', 'bar')
    image = self.db_api.image_get(self.adm_context, self.image['id'])
    self.assertEqual({'speed': '88mph', 'test_property': 'bar'}, self._propdict(image['properties']))
    self.assertOnlyImageHasProp(self.image['id'], 'test_property', 'bar')