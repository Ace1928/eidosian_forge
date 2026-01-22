from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def compare_nodelists(l1, l2):
    l1.sort()
    l2.sort()
    return l1 == l2