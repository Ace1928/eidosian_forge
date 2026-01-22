import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
def ex_get_permissions(self, container_name, object_name=None):
    """
        Return the permissions for the currently authenticated user.

        :param container_name: The container name.
        :type container_name: ``str``

        :param object_name: The object name. Optional. Not providing an object
            will return only container permissions.
        :type object_name: ``str`` or ``None``

        :return: A tuple of container and object permissions.
        :rtype: ``tuple`` of (``int``, ``int`` or ``None``) from
            ContainerPermissions and ObjectPermissions, respectively.
        """
    object_name = _clean_object_name(object_name)
    obj_perms = self._get_object_permissions(container_name, object_name) if object_name else None
    return (self._get_container_permissions(container_name), obj_perms)