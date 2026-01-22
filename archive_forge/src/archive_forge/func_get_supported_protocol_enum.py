from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_supported_protocol_enum(self, supported_protocol):
    """Get the supported_protocol enum.
             :param supported_protocol: The supported_protocol string
             :return: supported_protocol enum
        """
    supported_protocol = 'MULTI_PROTOCOL' if supported_protocol == 'MULTIPROTOCOL' else supported_protocol
    if supported_protocol in utils.FSSupportedProtocolEnum.__members__:
        return utils.FSSupportedProtocolEnum[supported_protocol]
    else:
        errormsg = 'Invalid choice {0} for supported_protocol'.format(supported_protocol)
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)