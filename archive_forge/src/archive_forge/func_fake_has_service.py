import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def fake_has_service(*args, **kwargs):
    return self.has_neutron