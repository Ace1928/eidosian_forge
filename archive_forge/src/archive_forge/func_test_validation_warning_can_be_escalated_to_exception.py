import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
def test_validation_warning_can_be_escalated_to_exception(self):
    warnings.filterwarnings(action='error')
    self.assertRaises(FutureWarning, self.field.coerce, 'obj', 'attr', 'not a uuid')