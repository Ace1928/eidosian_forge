import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def ex_get_hypervisor_hostname(self):
    """
        Return a system hostname on which the hypervisor is running.
        """
    hostname = self.connection.getHostname()
    return hostname