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
def test_image_destroy(self):
    location_data = [{'url': 'a', 'metadata': {'key': 'value'}, 'status': 'active'}, {'url': 'b', 'metadata': {}, 'status': 'active'}]
    fixture = {'status': 'queued', 'locations': location_data}
    image = self.db_api.image_create(self.context, fixture)
    IMG_ID = image['id']
    fixture = {'name': 'ping', 'value': 'pong', 'image_id': IMG_ID}
    prop = self.db_api.image_property_create(self.context, fixture)
    TENANT2 = str(uuid.uuid4())
    fixture = {'image_id': IMG_ID, 'member': TENANT2, 'can_share': False}
    member = self.db_api.image_member_create(self.context, fixture)
    self.db_api.image_tag_create(self.context, IMG_ID, 'snarf')
    self.assertEqual(2, len(image['locations']))
    self.assertIn('id', image['locations'][0])
    self.assertIn('id', image['locations'][1])
    image['locations'][0].pop('id')
    image['locations'][1].pop('id')
    self.assertEqual(location_data, image['locations'])
    self.assertEqual(('ping', 'pong', IMG_ID, False), (prop['name'], prop['value'], prop['image_id'], prop['deleted']))
    self.assertEqual((TENANT2, IMG_ID, False), (member['member'], member['image_id'], member['can_share']))
    self.assertEqual(['snarf'], self.db_api.image_tag_get_all(self.context, IMG_ID))
    image = self.db_api.image_destroy(self.adm_context, IMG_ID)
    self.assertTrue(image['deleted'])
    self.assertTrue(image['deleted_at'])
    self.assertRaises(exception.NotFound, self.db_api.image_get, self.context, IMG_ID)
    self.assertEqual([], image['locations'])
    prop = image['properties'][0]
    self.assertEqual(('ping', IMG_ID, True), (prop['name'], prop['image_id'], prop['deleted']))
    self.context.auth_token = 'user:%s:user' % TENANT2
    members = self.db_api.image_member_find(self.context, IMG_ID)
    self.assertEqual([], members)
    tags = self.db_api.image_tag_get_all(self.context, IMG_ID)
    self.assertEqual([], tags)