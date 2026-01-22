import os
import tempfile
from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import huawei
from os_brick.tests.initiator import test_connector
def test__analyze_output(self):
    cliout = 'ret_code=0\ndev_addr=/dev/vdxxx\nret_desc="success"'
    analyze_result = {'dev_addr': '/dev/vdxxx', 'ret_desc': '"success"', 'ret_code': '0'}
    result = self.connector._analyze_output(cliout)
    self.assertEqual(analyze_result, result)