from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def get_executions_representation(self, config, realm='master'):
    """
        Get a representation of the executions for an authentication flow.
        :param config: Representation of the authentication flow
        :param realm: Realm
        :return: Representation of the executions
        """
    try:
        executions = json.load(open_url(URL_AUTHENTICATION_FLOW_EXECUTIONS.format(url=self.baseurl, realm=realm, flowalias=quote(config['alias'], safe='')), method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
        for execution in executions:
            if 'authenticationConfig' in execution:
                execConfigId = execution['authenticationConfig']
                execConfig = json.load(open_url(URL_AUTHENTICATION_CONFIG.format(url=self.baseurl, realm=realm, id=execConfigId), method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs))
                execution['authenticationConfig'] = execConfig
        return executions
    except Exception as e:
        self.fail_open_url(e, msg='Could not get executions for authentication flow %s in realm %s: %s' % (config['alias'], realm, str(e)))