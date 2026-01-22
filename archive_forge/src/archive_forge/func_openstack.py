from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from zunclient.tests.functional import base
def openstack(self, *args, **kwargs):
    return self._zun(*args, cmd='openstack', **kwargs)