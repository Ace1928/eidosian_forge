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
def test_get_marker_obj(self):
    plugin = mock.Mock()
    plugin._get_myr.return_value = 'obj'
    obj = utils.get_marker_obj(plugin, 'ctx', 'myr', 10, mock.ANY)
    self.assertEqual('obj', obj)
    plugin._get_myr.assert_called_once_with('ctx', mock.ANY)