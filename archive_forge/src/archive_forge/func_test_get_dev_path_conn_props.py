import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data({'con_props': {'device_path': '/dev/sda'}, 'dev_info': {'path': None}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {'path': None}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': {'path': ''}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {'path': ''}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': {}}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': {}}, {'con_props': {'device_path': '/dev/sda'}, 'dev_info': None}, {'con_props': {'device_path': b'/dev/sda'}, 'dev_info': None})
@ddt.unpack
def test_get_dev_path_conn_props(self, con_props, dev_info):
    self.assertEqual('/dev/sda', utils.get_dev_path(con_props, dev_info))