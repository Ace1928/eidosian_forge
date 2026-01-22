from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def nodeinfo(self):
    self.__get_conn()
    data = self.conn.nodeinfo()
    info = dict(cpumodel=str(data[0]), phymemory=str(data[1]), cpus=str(data[2]), cpumhz=str(data[3]), numanodes=str(data[4]), sockets=str(data[5]), cpucores=str(data[6]), cputhreads=str(data[7]))
    return info