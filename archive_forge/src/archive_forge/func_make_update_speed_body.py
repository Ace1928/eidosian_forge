from __future__ import absolute_import, division, print_function
import re
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def make_update_speed_body(self, target_iface):
    target_iface = target_iface['iscsi']
    if self.speed is None:
        return (False, dict())
    else:
        if target_iface['interfaceData']['ethernetData']['autoconfigSupport']:
            self.module.warn("This interface's HIC speed is autoconfigured!")
            return (False, dict())
        if self.speed == strip_interface_speed(target_iface['interfaceData']['ethernetData']['currentInterfaceSpeed']):
            return (False, dict())
    supported_speeds = dict()
    for supported_speed in target_iface['interfaceData']['ethernetData']['supportedInterfaceSpeeds']:
        supported_speeds.update({strip_interface_speed(supported_speed): supported_speed})
    if self.speed not in supported_speeds:
        self.module.fail_json(msg='The host interface card (HIC) does not support the provided speed. Array Id [%s]. Supported speeds [%s]' % (self.ssid, ', '.join(supported_speeds.keys())))
    body = {'settings': {'maximumInterfaceSpeed': [supported_speeds[self.speed]]}, 'portsRef': {}}
    hic_ref = self.get_host_board_id(target_iface['id'])
    if hic_ref == '0000000000000000000000000000000000000000':
        body.update({'portsRef': {'portRefType': 'baseBoard', 'baseBoardRef': target_iface['id'], 'hicRef': ''}})
    else:
        body.update({'portsRef': {'portRefType': 'hic', 'hicRef': hic_ref, 'baseBoardRef': ''}})
    return (True, body)