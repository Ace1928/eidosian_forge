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
def test_image_update_ignores_atomics(self):
    image = self.db_api.image_get_all(self.adm_context)[0]
    self.db_api.image_set_property_atomic(image['id'], 'test1', 'foo')
    self.db_api.image_set_property_atomic(image['id'], 'test2', 'bar')
    self.db_api.image_update(self.adm_context, image['id'], {'properties': {'test1': 'baz', 'test3': 'bat', 'test4': 'yep'}}, purge_props=True, atomic_props=['test1', 'test2', 'test3'])
    image = self.db_api.image_get(self.adm_context, image['id'])
    self.assertEqual({'test1': 'foo', 'test2': 'bar', 'test4': 'yep'}, self._propdict(image['properties']))