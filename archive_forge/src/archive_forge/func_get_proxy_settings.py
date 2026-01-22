from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def get_proxy_settings(scheme='https'):
    """ Returns a tuple containing (proxy_host, proxy_port). (False, False) if no proxy is found """
    proxy_url = getproxies().get(scheme, '')
    if not proxy_url:
        return (False, False)
    else:
        parsed_url = urlparse(proxy_url)
        if parsed_url.scheme:
            proxy_host, proxy_port = parsed_url.netloc.split(':')
        else:
            proxy_host, proxy_port = parsed_url.path.split(':')
        return (proxy_host, proxy_port)