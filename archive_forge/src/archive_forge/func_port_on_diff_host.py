from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def port_on_diff_host(self, arg_port):
    """ Checks to see if a passed in port arg is present on a different host"""
    for host in self.all_hosts:
        if host['name'].lower() != self.name.lower():
            for port in host['hostSidePorts']:
                if arg_port['label'].lower() == port['label'].lower() or arg_port['port'].lower() == port['address'].lower():
                    return True
    return False