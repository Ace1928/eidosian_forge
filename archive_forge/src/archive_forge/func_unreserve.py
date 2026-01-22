from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def unreserve(self, session):
    """Unreserve volume"""
    body = {'os-unreserve': None}
    self._action(session, body)