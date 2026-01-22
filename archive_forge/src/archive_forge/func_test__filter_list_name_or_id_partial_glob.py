from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test__filter_list_name_or_id_partial_glob(self):
    el1 = dict(id=100, name='donald')
    el2 = dict(id=200, name='pluto[2017-01-10]')
    data = [el1, el2]
    ret = _utils._filter_list(data, 'pluto*', None)
    self.assertEqual([el2], ret)