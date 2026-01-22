from unittest import mock
from oslo_config import cfg
from oslo_db.sqlalchemy import models
import sqlalchemy as sa
from sqlalchemy.ext import declarative
from sqlalchemy import orm
from neutron_lib.api import attributes
from neutron_lib import context
from neutron_lib.db import utils
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
def test_model_query_scope_is_project_admin(self):
    ctx = context.Context(project_id='some project', is_admin=True, is_advsvc=False)
    model = mock.Mock(project_id='project')
    self.assertFalse(utils.model_query_scope_is_project(ctx, model))
    del model.project_id
    self.assertFalse(utils.model_query_scope_is_project(ctx, model))