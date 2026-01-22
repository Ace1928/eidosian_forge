import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def exec_on_ctn(self, cmd, capture=True, detach=False):
    name = self.docker_name()
    flag = '-d' if detach else ''
    return self.dcexec('docker exec {0} {1} {2}'.format(flag, name, cmd), capture=capture)