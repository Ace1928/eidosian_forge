import os
import platform
import random
import time
import netaddr
from neutron_lib.utils import helpers
from neutron_lib.utils import net
def reset_random_seed():
    seed = time.time() + os.getpid()
    random.seed(seed)