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
def test_resource_get_all_by_with_admin_context(self):
    admin_ctx = utils.dummy_context(is_admin=True, tenant_id='admin_tenant')
    create_resource(self.ctx, self.stack, phys_res_id=UUID1)
    create_resource(self.ctx, self.stack, phys_res_id=UUID2)
    ret_res = db_api.resource_get_all_by_physical_resource_id(admin_ctx, UUID1)
    ret_list = list(ret_res)
    self.assertEqual(1, len(ret_list))
    self.assertEqual(UUID1, ret_list[0].physical_resource_id)
    mt = db_api.resource_get_all_by_physical_resource_id(admin_ctx, UUID2)
    ret_list = list(mt)
    self.assertEqual(1, len(ret_list))
    self.assertEqual(UUID2, ret_list[0].physical_resource_id)