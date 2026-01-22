import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
def test__get_invalid_resource_id_raises(self):
    resource_ids = [[], {}, False, '', 0, None, ()]
    for resource_id in resource_ids:
        self.assertRaises(exc.ValidationError, self.manager._get, resource_id=resource_id)