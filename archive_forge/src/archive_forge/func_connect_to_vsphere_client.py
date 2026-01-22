from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def connect_to_vsphere_client(self):
    """
        Connect to vSphere API Client with Username and Password

        """
    username = self.params.get('username')
    password = self.params.get('password')
    hostname = self.params.get('hostname')
    validate_certs = self.params.get('validate_certs')
    port = self.params.get('port')
    session = requests.Session()
    session.verify = validate_certs
    protocol = self.params.get('protocol')
    proxy_host = self.params.get('proxy_host')
    proxy_port = self.params.get('proxy_port')
    if validate_certs is False:
        if HAS_URLLIB3:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    if all([protocol, proxy_host, proxy_port]):
        proxies = {protocol: '{0}://{1}:{2}'.format(protocol, proxy_host, proxy_port)}
        session.proxies.update(proxies)
    if not all([hostname, username, password]):
        self.module.fail_json(msg='Missing one of the following : hostname, username, password. Please read the documentation for more information.')
    msg = 'Failed to connect to vCenter or ESXi API at %s:%s' % (hostname, port)
    try:
        client = create_vsphere_client(server='%s:%s' % (hostname, port), username=username, password=password, session=session)
    except requests.exceptions.SSLError as ssl_exc:
        msg += ' due to SSL verification failure'
        self.module.fail_json(msg='%s : %s' % (msg, to_native(ssl_exc)))
    except Exception as generic_exc:
        self.module.fail_json(msg='%s : %s' % (msg, to_native(generic_exc)))
    if client is None:
        self.module.fail_json(msg='Failed to login to %s' % hostname)
    return client