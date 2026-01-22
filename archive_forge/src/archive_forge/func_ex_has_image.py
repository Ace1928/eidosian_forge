import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def ex_has_image(self, alias):
    """
        Helper function. Returns true and the image fingerprint
        if the image with the given alias exists on the host.

        :param alias: the image alias
        :type  alias: ``str``

        :rtype:  ``tupple`` :: (``boolean``, ``str``)
        """
    try:
        response = self.connection.request('/{}/images/aliases/{}'.format(self.version, alias))
        metadata = response.object['metadata']
        return (True, metadata.get('target'))
    except BaseHTTPError as err:
        lxd_exception = self._get_lxd_api_exception_for_error(err)
        if lxd_exception.message == 'not found':
            return (False, -1)
        else:
            raise lxd_exception
    except Exception as err:
        raise self._get_lxd_api_exception_for_error(err)