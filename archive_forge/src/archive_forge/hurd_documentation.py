from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.network.base import Network, NetworkCollector

    This is a GNU Hurd specific subclass of Network. It use fsysopts to
    get the ip address and support only pfinet.
    