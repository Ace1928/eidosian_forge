from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def nagios_cmd(self, cmd):
    """
        This sends an arbitrary command to nagios

        It prepends the submitted time and appends a 


        You just have to provide the properly formatted command
        """
    pre = '[%s]' % int(time.time())
    post = '\n'
    cmdstr = '%s %s%s' % (pre, cmd, post)
    self._write_command(cmdstr)