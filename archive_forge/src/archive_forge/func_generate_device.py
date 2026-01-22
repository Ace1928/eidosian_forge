import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
def generate_device(self, location, iqn, transport=None, lun=1):
    dev_format = 'ip-%s-iscsi-%s-lun-%s' % (location, iqn, lun)
    if transport:
        dev_format = 'pci-0000:00:00.0-' + dev_format
    fake_dev_path = '/dev/disk/by-path/' + dev_format
    return fake_dev_path