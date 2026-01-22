import unittest
from subprocess import Popen, PIPE, STDOUT
import time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
def test_of_config(self):
    self.skipTest('OVS 1.10 does not support of_config')