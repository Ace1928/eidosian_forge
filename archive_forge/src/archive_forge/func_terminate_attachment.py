from openstack.common import metadata
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack import utils
def terminate_attachment(self, session, connector):
    """Terminate volume attachment"""
    body = {'os-terminate_connection': {'connector': connector}}
    self._action(session, body)