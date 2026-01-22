import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_reachablily(self, prefix, timeout=20):
    version = netaddr.IPNetwork(prefix).version
    addr = prefix.split('/')[0]
    if version == 4:
        ping_cmd = 'ping'
    elif version == 6:
        ping_cmd = 'ping6'
    else:
        raise Exception('unsupported route family: {0}'.format(version))
    cmd = '/bin/bash -c "/bin/{0} -c 1 -w 1 {1} | xargs echo"'.format(ping_cmd, addr)
    interval = 1
    count = 0
    while True:
        res = self.exec_on_ctn(cmd)
        LOG.info(res)
        if '1 packets received' in res and '0% packet loss':
            break
        time.sleep(interval)
        count += interval
        if count >= timeout:
            raise Exception('timeout')
    return True