from __future__ import absolute_import, division, print_function
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
from random import randint
import time
def get_sms_connection(self):
    """
        Creates a Service instance for VMware SMS
        """
    client_stub = self.si._GetStub()
    try:
        session_cookie = client_stub.cookie.split('"')[1]
    except IndexError:
        self.module.fail_json(msg='Failed to get session cookie')
    ssl_context = client_stub.schemeArgs.get('context')
    additional_headers = {'vcSessionCookie': session_cookie}
    hostname = self.module.params['hostname']
    if not hostname:
        self.module.fail_json(msg='Please specify required parameter - hostname')
    stub = SoapStubAdapter(host=hostname, path='/sms/sdk', version=self.version, sslContext=ssl_context, requestContext=additional_headers)
    self.sms_si = sms.ServiceInstance('ServiceInstance', stub)