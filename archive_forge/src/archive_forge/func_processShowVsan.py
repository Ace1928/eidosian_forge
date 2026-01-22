from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def processShowVsan(self):
    patv = '^vsan\\s+(\\d+)\\s+information'
    patnamestate = 'name:(.*)state:(.*)'
    patoperstate = 'operational state:(.*)'
    output = self.execute_show_vsan_cmd().split('\n')
    for o in output:
        z = re.match(patv, o.strip())
        if z:
            v = z.group(1).strip()
            self.vsaninfo[v] = Vsan(v)
        z1 = re.match(patnamestate, o.strip())
        if z1:
            n = z1.group(1).strip()
            s = z1.group(2).strip()
            self.vsaninfo[v].vsanname = n
            self.vsaninfo[v].vsanstate = s
        z2 = re.match(patoperstate, o.strip())
        if z2:
            oper = z2.group(1).strip()
            self.vsaninfo[v].vsanoperstate = oper
    self.vsaninfo['4079'] = Vsan('4079')
    self.vsaninfo['4094'] = Vsan('4094')