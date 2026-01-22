from openstack.tests import fakes
from openstack.tests.unit import base
def get_fake_has_service(has_service):

    def fake_has_service(s):
        if s == 'network':
            return False
        return has_service(s)
    return fake_has_service