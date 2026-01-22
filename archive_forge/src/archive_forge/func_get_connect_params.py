from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def get_connect_params(auth, fail_function):
    if is_using_tls(auth):
        auth['docker_host'] = auth['docker_host'].replace('tcp://', 'https://')
    result = dict(base_url=auth['docker_host'], version=auth['api_version'], timeout=auth['timeout'])
    if auth['tls_verify']:
        tls_config = dict(verify=True, assert_hostname=auth['tls_hostname'], ssl_version=auth['ssl_version'], fail_function=fail_function)
        if auth['cert_path'] and auth['key_path']:
            tls_config['client_cert'] = (auth['cert_path'], auth['key_path'])
        if auth['cacert_path']:
            tls_config['ca_cert'] = auth['cacert_path']
        result['tls'] = _get_tls_config(**tls_config)
    elif auth['tls']:
        tls_config = dict(verify=False, ssl_version=auth['ssl_version'], fail_function=fail_function)
        if auth['cert_path'] and auth['key_path']:
            tls_config['client_cert'] = (auth['cert_path'], auth['key_path'])
        result['tls'] = _get_tls_config(**tls_config)
    if auth.get('use_ssh_client'):
        if LooseVersion(docker_version) < LooseVersion('4.4.0'):
            fail_function('use_ssh_client=True requires Docker SDK for Python 4.4.0 or newer')
        result['use_ssh_client'] = True
    return result