from openstack import exceptions
from openstack import resource
from openstack import utils
def thaw(self, session):
    body = {'host': self.host}
    return self._action(session, 'thaw', body)