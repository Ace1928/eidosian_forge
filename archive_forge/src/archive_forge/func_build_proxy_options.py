from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def build_proxy_options():
    """ Returns list of valid proxy options for keytool """
    proxy_host, proxy_port = get_proxy_settings()
    no_proxy = os.getenv('no_proxy')
    proxy_opts = []
    if proxy_host:
        proxy_opts.extend(['-J-Dhttps.proxyHost=%s' % proxy_host, '-J-Dhttps.proxyPort=%s' % proxy_port])
        if no_proxy is not None:
            non_proxy_hosts = no_proxy.replace(',', '|')
            non_proxy_hosts = re.sub('(^|\\|)\\.', '\\1*.', non_proxy_hosts)
            proxy_opts.extend(['-J-Dhttp.nonProxyHosts=%s' % non_proxy_hosts])
    return proxy_opts