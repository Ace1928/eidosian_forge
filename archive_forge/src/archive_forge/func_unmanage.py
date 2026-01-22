from openstack.common import metadata
from openstack import format
from openstack import resource
from openstack import utils
def unmanage(self, session):
    """Unmanage volume"""
    body = {'os-unmanage': None}
    self._action(session, body)