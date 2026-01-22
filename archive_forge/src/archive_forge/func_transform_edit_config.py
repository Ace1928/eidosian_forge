from .default import DefaultDeviceHandler
from ncclient.operations.third_party.iosxe.rpc import SaveConfig
from ncclient.xml_ import BASE_NS_1_0
import logging
def transform_edit_config(self, node):
    nodes = node.findall('./config')
    if len(nodes) == 1:
        logger.debug('IOS XE handler: patching namespace of config element')
        nodes[0].tag = '{%s}%s' % (BASE_NS_1_0, 'config')
    return node