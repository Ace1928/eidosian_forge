from openstack.clustering.v1 import _async_resource
from openstack.common import metadata
from openstack import resource
from openstack import utils
def scale_out(self, session, count=None):
    body = {'scale_out': {'count': count}}
    return self.action(session, body)