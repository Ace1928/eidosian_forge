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
@mock.patch.object(attributes, 'populate_project_info')
def test_resource_fields(self, mock_populate):
    r = {'name': 'n', 'id': '1', 'desc': None}
    utils.resource_fields(r, ['name'])
    mock_populate.assert_called_once_with({'name': 'n'})