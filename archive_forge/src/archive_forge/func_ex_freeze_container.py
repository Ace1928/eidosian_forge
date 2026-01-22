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
def ex_freeze_container(self, container, ex_timeout=default_time_out):
    """
        Set the given container into a freeze state

        :param container: The container to restart
        :type  container: :class:`.Container`

        :param ex_timeout: Time to wait for the operation to complete
        :type  ex_timeout: ``int``

        :rtype :class: `libcloud.container.base.Container
        """
    return self._do_container_action(container=container, action='freeze', timeout=ex_timeout, force=True, stateful=True)