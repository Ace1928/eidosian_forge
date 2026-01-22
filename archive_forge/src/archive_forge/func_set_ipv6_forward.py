import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def set_ipv6_forward(self):
    cmd = 'sysctl -w net.ipv6.conf.all.forwarding=1'
    self.exec_on_ctn(cmd)