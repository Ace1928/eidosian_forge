import itertools
import logging
import os
import re
import subprocess
import time
import netaddr
def get_bridges(self):
    if self.br_type == BRIDGE_TYPE_DOCKER:
        return self.get_bridges_dc()
    elif self.br_type == BRIDGE_TYPE_BRCTL:
        return self.get_bridges_brctl()
    elif self.br_type == BRIDGE_TYPE_OVS:
        return self.get_bridges_ovs()