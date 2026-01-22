from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.snmp_server.snmp_server import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.snmp_server import (
def get_snmpv3_user_data(self, connection):
    """get snmpv3 user data from the device

        :param connection: the device connection

        :rtype: string
        :returns: snmpv3 user data

        Note: The seperate method is needed because the snmpv3 user data is not returned within the snmp-server config
        """
    try:
        _get_snmpv3_user = connection.get('show snmp user')
    except Exception as e:
        if 'agent not enabled' in str(e):
            return ''
        raise Exception('Unable to get snmp user data: %s' % str(e))
    return _get_snmpv3_user