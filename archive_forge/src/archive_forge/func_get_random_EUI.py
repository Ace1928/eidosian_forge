import os
import platform
import random
import time
import netaddr
from neutron_lib.utils import helpers
from neutron_lib.utils import net
def get_random_EUI():
    return netaddr.EUI(net.get_random_mac(['fe', '16', '3e', '00', '00', '00']))