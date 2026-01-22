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
def test_paginate_non_redundant_sort_keys(self):
    original_method = self.db_api._paginate_query

    def fake_paginate_query(query, model, limit, sort_keys, marker, sort_dir, sort_dirs):
        self.assertEqual(['name', 'created_at', 'id'], sort_keys)
        return original_method(query, model, limit, sort_keys, marker, sort_dir, sort_dirs)
    self.mock_object(self.db_api, '_paginate_query', fake_paginate_query)
    self.db_api.image_get_all(self.context, sort_key=['name'])