import os
import platform
import random
import time
import netaddr
from neutron_lib.utils import helpers
from neutron_lib.utils import net
def get_random_ip_network(version=4):
    return netaddr.IPNetwork(get_random_cidr(version=version))