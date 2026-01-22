from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def fake_pretend_lvm_version(obj, *cmd, **kwargs):
    return ('  LVM version:     2.03.00 (2012-03-06)\n', '')