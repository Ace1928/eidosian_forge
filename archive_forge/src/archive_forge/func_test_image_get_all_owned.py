import copy
import datetime
import functools
from unittest import mock
import uuid
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import utils as sqlalchemyutils
from sqlalchemy import sql
from glance.common import exception
from glance.common import timeutils
from glance import context
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import models
from glance.tests import functional
import glance.tests.functional.db as db_tests
from glance.tests import utils as test_utils
def test_image_get_all_owned(self):
    TENANT1 = str(uuid.uuid4())
    ctxt1 = context.RequestContext(is_admin=False, tenant=TENANT1, auth_token='user:%s:user' % TENANT1)
    UUIDX = str(uuid.uuid4())
    image_meta_data = {'id': UUIDX, 'status': 'queued', 'owner': TENANT1}
    self.db_api.image_create(ctxt1, image_meta_data)
    TENANT2 = str(uuid.uuid4())
    ctxt2 = context.RequestContext(is_admin=False, tenant=TENANT2, auth_token='user:%s:user' % TENANT2)
    UUIDY = str(uuid.uuid4())
    image_meta_data = {'id': UUIDY, 'status': 'queued', 'owner': TENANT2}
    self.db_api.image_create(ctxt2, image_meta_data)
    images = self.db_api.image_get_all(ctxt1)
    image_ids = [image['id'] for image in images]
    expected = [UUIDX, UUID3, UUID2, UUID1]
    self.assertEqual(sorted(expected), sorted(image_ids))