from unittest import mock
from os_brick import exception
from os_brick.initiator.connectors import storpool as connector
from os_brick.tests.initiator import test_connector
def volumeName(self, vid):
    return volumeNameExt(vid)