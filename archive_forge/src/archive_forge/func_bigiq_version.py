from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def bigiq_version(client):
    uri = 'https://{0}:{1}/mgmt/shared/resolver/device-groups/cm-shared-all-big-iqs/devices'.format(client.provider['server'], client.provider['server_port'])
    query = '?$select=version'
    resp = client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 403]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    if 'items' in response:
        version = response['items'][0]['version']
        return version
    raise F5ModuleError('Failed to retrieve BIG-IQ version information.')