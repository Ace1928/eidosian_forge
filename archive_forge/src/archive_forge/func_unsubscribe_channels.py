from __future__ import absolute_import, division, print_function
import ssl
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xmlrpc_client
def unsubscribe_channels(channelname, client, session, sysname, sys_id):
    channels = base_channels(client, session, sys_id)
    channels.remove(channelname)
    return client.system.setChildChannels(session, sys_id, channels)