from unittest import mock
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware.tests import base
def test_log_exception_to_string(self):
    self.assertEqual('Insufficient disk space.', str(exceptions.NoDiskSpaceException()))