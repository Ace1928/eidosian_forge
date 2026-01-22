import logging
import os
import sys
import unittest
from os_ken.lib import pcaplib
from os_ken.lib.packet import gre
from os_ken.lib.packet import packet
from os_ken.utils import binary_str
from os_ken.lib.packet.ether_types import ETH_TYPE_IP, ETH_TYPE_TEB

    Test case gre for os_ken.lib.packet.gre.
    