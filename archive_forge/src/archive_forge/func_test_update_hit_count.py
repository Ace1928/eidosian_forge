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
def test_update_hit_count(self):
    hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
    self.assertEqual(3, hit_count)
    self.db_api.update_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
    hit_count = self.db_api.get_hit_count(self.adm_context, self.images[0]['id'], 'node_url_1')
    self.assertEqual(4, hit_count)