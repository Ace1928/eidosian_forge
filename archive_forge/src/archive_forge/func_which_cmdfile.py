from __future__ import absolute_import, division, print_function
import time
import os.path
import stat
from ansible.module_utils.basic import AnsibleModule
def which_cmdfile():
    locations = ['/etc/nagios/nagios.cfg', '/etc/nagios3/nagios.cfg', '/etc/nagios2/nagios.cfg', '/usr/local/etc/nagios/nagios.cfg', '/usr/local/groundwork/nagios/etc/nagios.cfg', '/omd/sites/oppy/tmp/nagios/nagios.cfg', '/usr/local/nagios/etc/nagios.cfg', '/usr/local/nagios/nagios.cfg', '/opt/nagios/etc/nagios.cfg', '/opt/nagios/nagios.cfg', '/etc/icinga/icinga.cfg', '/usr/local/icinga/etc/icinga.cfg']
    for path in locations:
        if os.path.exists(path):
            for line in open(path):
                if line.startswith('command_file'):
                    return line.split('=')[1].strip()
    return None