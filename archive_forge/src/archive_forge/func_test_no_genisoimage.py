import json
import os
from unittest import mock
import testtools
from openstack.baremetal import configdrive
def test_no_genisoimage(self, mock_popen):
    mock_popen.side_effect = OSError
    self.assertRaisesRegex(RuntimeError, 'genisoimage', configdrive.pack, '/fake')