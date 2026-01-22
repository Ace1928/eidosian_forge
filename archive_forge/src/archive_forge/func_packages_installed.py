from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def packages_installed(client):
    """Returns a list of installed ATC packages

    Args:
        client: Client connection to the BIG-IP

    Returns:
        A list of installed packages in their short name for.
        For example, ['as3', 'do', 'ts']
    """
    packages = {'f5-declarative-onboarding': 'do', 'f5-appsvcs': 'as3', 'f5-appsvcs-templates': 'fast', 'f5-cloud-failover': 'cfe', 'f5-telemetry': 'ts', 'f5-service-discovery': 'sd'}
    uri = 'https://{0}:{1}/mgmt/shared/iapp/global-installed-packages'.format(client.provider['server'], client.provider['server_port'])
    resp = client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] == 404:
        return []
    if resp.status not in [200, 201] or ('code' in response and response['code'] not in [200, 201]):
        raise F5ModuleError(resp.content)
    if 'items' not in response:
        return []
    result = [packages[x['appName']] for x in response['items'] if x['appName'] in packages.keys()]
    return result