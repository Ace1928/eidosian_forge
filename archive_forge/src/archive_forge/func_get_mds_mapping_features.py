from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def get_mds_mapping_features():
    feature_to_be_mapped = {'show': {'fcrxbbcredit': 'extended_credit', 'port-track': 'port_track', 'scp-server': 'scpServer', 'sftp-server': 'sftpServer', 'ssh': 'sshServer', 'tacacs+': 'tacacs', 'telnet': 'telnetServer'}, 'config': {'extended_credit': 'fcrxbbcredit', 'port_track': 'port-track', 'scpServer': 'scp-server', 'sftpServer': 'sftp-server', 'sshServer': 'ssh', 'tacacs': 'tacacs+', 'telnetServer': 'telnet'}}
    return feature_to_be_mapped