import warnings
from openstack import exceptions
from openstack import resource
from openstack import utils
from openstack import warnings as os_warnings
def get_uptime(self, session):
    """Get uptime information for the hypervisor

        Updates uptime attribute of the hypervisor object
        """
    warnings.warn('This call is deprecated and is only available until Nova 2.88', os_warnings.LegacyAPIWarning)
    if utils.supports_microversion(session, '2.88'):
        raise exceptions.SDKException('Hypervisor.get_uptime is not supported anymore')
    url = utils.urljoin(self.base_path, self.id, 'uptime')
    microversion = self._get_microversion(session, action='fetch')
    response = session.get(url, microversion=microversion)
    self._translate_response(response)
    return self