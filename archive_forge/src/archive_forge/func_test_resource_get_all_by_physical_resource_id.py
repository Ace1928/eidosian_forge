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
def test_resource_get_all_by_physical_resource_id(self):
    create_resource(self.ctx, self.stack)
    create_resource(self.ctx, self.stack)
    ret_res = db_api.resource_get_all_by_physical_resource_id(self.ctx, UUID1)
    ret_list = list(ret_res)
    self.assertEqual(2, len(ret_list))
    for res in ret_list:
        self.assertEqual(UUID1, res.physical_resource_id)
    mt = db_api.resource_get_all_by_physical_resource_id(self.ctx, UUID2)
    self.assertFalse(list(mt))