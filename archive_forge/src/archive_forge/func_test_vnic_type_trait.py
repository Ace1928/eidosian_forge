from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_vnic_type_trait(self):
    self.assertEqual('CUSTOM_VNIC_TYPE_SOMEVNICTYPE', place_utils.vnic_type_trait('somevnictype'))