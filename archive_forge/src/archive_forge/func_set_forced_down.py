from openstack import exceptions
from openstack import resource
from openstack import utils
def set_forced_down(self, session, host=None, binary=None, forced=False):
    """Update forced_down information of a service."""
    microversion = session.default_microversion
    body = {}
    if not host:
        host = self.host
    if not binary:
        binary = self.binary
    body = {'host': host, 'binary': binary}
    if utils.supports_microversion(session, '2.11'):
        body['forced_down'] = forced
        microversion = '2.11'
    return self._action(session, 'force-down', body, microversion=microversion)