import datetime
import json
import time
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_utils import timeutils
from sqlalchemy import orm
from sqlalchemy.orm import exc
from sqlalchemy.orm import session
from heat.common import context
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import resource as rsrc
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.engine import template_files
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_resource_data_delete(self):
    create_resource_data(self.ctx, self.resource)
    res_data = db_api.resource_data_get_by_key(self.ctx, self.resource.id, 'test_resource_key')
    self.assertIsNotNone(res_data)
    self.assertEqual('test_value', res_data.value)
    db_api.resource_data_delete(self.ctx, self.resource.id, 'test_resource_key')
    self.assertRaises(exception.NotFound, db_api.resource_data_get_by_key, self.ctx, self.resource.id, 'test_resource_key')
    self.assertIsNotNone(res_data)
    self.assertRaises(exception.NotFound, db_api.resource_data_get_all, self.resource.context, self.resource.id)