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
def test_node_reference_get_by_url(self):
    node_reference = self.db_api.node_reference_get_by_url(self.adm_context, 'node_url_1')
    self.assertEqual('node_url_1', node_reference['node_reference_url'])