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
def test_delete_create_delete(self):
    """Try to delete, re-create, and then re-delete property."""
    self.db_api.image_delete_property_atomic(self.image['id'], 'speed', '88mph')
    self.db_api.image_update(self.adm_context, self.image['id'], {'properties': {'speed': '89mph'}}, purge_props=True)
    self.assertRaises(exception.NotFound, self.db_api.image_delete_property_atomic, self.image['id'], 'speed', '88mph')
    self.db_api.image_delete_property_atomic(self.image['id'], 'speed', '89mph')