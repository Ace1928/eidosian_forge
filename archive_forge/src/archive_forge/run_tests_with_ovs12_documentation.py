import unittest
from subprocess import Popen, PIPE, STDOUT
import time
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
Set protocols parameter for OVS version 1.10