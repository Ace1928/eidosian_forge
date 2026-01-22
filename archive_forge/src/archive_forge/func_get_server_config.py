from libcloud.common.google import GoogleResponse, GoogleBaseConnection, GoogleOAuth2Credential
from libcloud.container.providers import Provider
from libcloud.container.drivers.kubernetes import KubernetesContainerDriver
def get_server_config(self, ex_zone=None):
    """
        Return configuration info about the Container Engine service.

        :keyword  ex_zone:  Optional zone name or None
        :type     ex_zone:  ``str`` or :class:`GCEZone` or
                            :class:`NodeLocation` or ``None``
        """
    if ex_zone is None:
        ex_zone = self.zone
    request = '/zones/%s/serverconfig' % ex_zone
    response = self.connection.request(request, method='GET').object
    return response