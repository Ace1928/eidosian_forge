import errno
import hashlib
import testtools
from unittest import mock
import ddt
from glanceclient.common import utils as common_utils
from glanceclient import exc
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import images
def test_image_tasks_not_supported(self):
    with mock.patch.object(common_utils, 'has_version') as mock_has_version:
        mock_has_version.return_value = False
        self.assertRaises(exc.HTTPNotImplemented, self.controller.get_associated_image_tasks, '3a4560a1-e585-443e-9b39-553b46ec92d1')