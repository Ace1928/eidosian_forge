import os
from unittest import mock
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_middleware import request_id
from oslo_policy import opts as policy_opts
from oslo_utils import importutils
import webob
from heat.common import context
from heat.common import exception
from heat.tests import common
def test_get_admin_context(self):
    ctx = context.get_admin_context()
    self.assertTrue(ctx.is_admin)
    self.assertFalse(ctx.show_deleted)